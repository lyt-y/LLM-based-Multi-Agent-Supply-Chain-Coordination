"""Per-run LLM resource and communication measurement."""

from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from time import perf_counter


@dataclass
class CoordinationMetrics:
    """One instance per run. Client calls include cache hits; retries are opaque."""

    decision_llm_call_count: int = 0
    decision_input_tokens: int = 0
    decision_output_tokens: int = 0
    coordination_llm_call_count: int = 0
    coordination_input_tokens: int = 0
    coordination_output_tokens: int = 0
    coordination_message_count: int = 0
    wall_clock_seconds: float = 0.0
    _missing_usage: dict = field(default_factory=lambda: {'decision': 0, 'coordination': 0})
    _links: dict = field(default_factory=dict)
    _rounds: set = field(default_factory=set)

    @staticmethod
    def _check_kind(kind):
        if kind not in ('decision', 'coordination'):
            raise ValueError('Usage kind must be decision or coordination')

    @staticmethod
    def _get(obj, key):
        return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)

    def record_call(self, kind, response=None):
        """Record one client invocation and authoritative response usage, if present."""
        self._check_kind(kind)
        name = f'{kind}_llm_call_count'
        setattr(self, name, getattr(self, name) + 1)
        usage = self._get(response, 'usage')
        incoming = self._get(usage, 'prompt_tokens')
        outgoing = self._get(usage, 'completion_tokens')
        if incoming is None:
            incoming = self._get(usage, 'input_tokens')
        if outgoing is None:
            outgoing = self._get(usage, 'output_tokens')
        # Missing/incomplete metadata is explicitly flagged, never tokenized.
        if any(not isinstance(n, int) or isinstance(n, bool) or n < 0 for n in (incoming, outgoing)):
            self._missing_usage[kind] += 1
            return
        for suffix, amount in [('input_tokens', incoming), ('output_tokens', outgoing)]:
            name = f'{kind}_{suffix}'
            setattr(self, name, getattr(self, name) + amount)

    def call(self, kind, function, *args, **kwargs):
        """Pass arguments, response identity and exceptions through unchanged."""
        self._check_kind(kind)
        try:
            response = function(*args, **kwargs)
        except BaseException:
            self.record_call(kind)
            raise
        self.record_call(kind, response)
        return response

    @contextmanager
    def track_client(self, client, kind='decision'):
        """Observe a synchronous AutoGen client.create; restore it on every exit.

        Use once per distinct client per run, without concurrent/nested wrapping.
        A future mixed decision/coordination protocol must label each call instead.
        """
        self._check_kind(kind)
        original = client.create
        had_instance_method = 'create' in vars(client)
        instance_method = vars(client).get('create')

        @wraps(original)
        def measured(*args, **kwargs):
            return self.call(kind, original, *args, **kwargs)

        client.create = measured
        try:
            yield
        finally:
            if had_instance_method:
                client.create = instance_method
            else:
                del client.create

    @contextmanager
    def measure_run(self):
        """Measure the run only, excluding setup done before entering this block."""
        start = perf_counter()
        try:
            yield self
        finally:
            self.wall_clock_seconds += perf_counter() - start

    def record_round(self, round_id):
        """Explicit protocol identifier, unique across the run (not inferred time)."""
        self._rounds.add(round_id)

    def record_message(self, sender, receiver, *, round_id=None):
        """Call once per actual directed agent-to-agent delivery, not local decisions."""
        if not isinstance(sender, str) or not isinstance(receiver, str) or sender == receiver:
            raise ValueError('Use distinct sender and receiver agent names')
        self.coordination_message_count += 1
        link = (sender, receiver)
        self._links[link] = self._links.get(link, 0) + 1
        if round_id is not None:
            self.record_round(round_id)

    def to_dict(self):
        result = {}
        for kind in ('decision', 'coordination'):
            for suffix in ('llm_call_count', 'input_tokens', 'output_tokens'):
                result[f'{kind}_{suffix}'] = getattr(self, f'{kind}_{suffix}')
            result[f'{kind}_total_tokens'] = result[f'{kind}_input_tokens'] + result[f'{kind}_output_tokens']
        for suffix in ('llm_call_count', 'input_tokens', 'output_tokens', 'total_tokens'):
            result[suffix] = result[f'decision_{suffix}'] + result[f'coordination_{suffix}']
        result.update(
            coordination_message_count=self.coordination_message_count,
            coordination_round_count=len(self._rounds),
            active_communication_links=len(self._links),
            per_link_message_counts=[dict(sender=s, receiver=r, message_count=n)
                                     for (s, r), n in sorted(self._links.items())],
            wall_clock_seconds=self.wall_clock_seconds,
            usage_unavailable_call_count=sum(self._missing_usage.values()),
            decision_usage_unavailable_call_count=self._missing_usage['decision'],
            coordination_usage_unavailable_call_count=self._missing_usage['coordination'],
        )
        return result

    def add_to_result(self, result):
        """Attach a JSON-safe snapshot without replacing any legacy field."""
        if 'coordination_metrics' in result:
            raise ValueError('Result already contains coordination_metrics')
        result['coordination_metrics'] = self.to_dict()
        return result

    def restore(self, snapshot, round_ids):
        """Restore observed resources without recounting replayed physical periods."""
        if self.decision_llm_call_count or self.coordination_llm_call_count:
            raise ValueError('Resume requires fresh metrics')
        for kind in ('decision', 'coordination'):
            for suffix in ('llm_call_count', 'input_tokens', 'output_tokens'):
                name = f'{kind}_{suffix}'
                value = snapshot[name]
                if type(value) is not int or value < 0:
                    raise ValueError('Invalid checkpoint resource counter')
                setattr(self, name, value)
            self._missing_usage[kind] = snapshot[f'{kind}_usage_unavailable_call_count']
        self.coordination_message_count = snapshot['coordination_message_count']
        self.wall_clock_seconds = snapshot['wall_clock_seconds']
        self._links = {(row['sender'], row['receiver']): row['message_count']
                       for row in snapshot['per_link_message_counts']}
        self._rounds = set(round_ids)


