"""«Чтобы довести до конца» (владелец 28.09.2026, brain-desk #904): неполный
результат заканчивается 1–3 действиями владельца, а не перечнем пробелов."""
from deep_agent import msty_prompts as prompts


def test_close_gaps_block_in_policy():
    policy = prompts.POLICY
    assert 'MSTY_CLOSE_GAPS_V1.' in policy
    assert '«Чтобы довести до конца:»' in policy
    assert '1–3 конкретных действий' in policy
    assert 'owner_browser_allow_site' in policy
    assert 'пароли\nи коды вводит владелец' in policy
    assert 'назови недостающий tool, данные, решение или лимит' not in policy


def test_close_gaps_kept_whenever_an_external_tool_is_on_the_wire():
    for intent in ('direct', 'read', 'mutate'):
        for domains in ([], ['tax']):
            route = {'intent': intent, 'domains': domains}
            kept = prompts.select_policy(prompts.POLICY, route, ('read_file',))
            assert 'MSTY_CLOSE_GAPS_V1.' in kept, (intent, domains)
            # No tools: nothing to call; the short form stays in continuity.
            bare = prompts.select_policy(prompts.POLICY, route, (), external_names=())
            assert 'MSTY_CLOSE_GAPS_V1.' not in bare
            assert '«Чтобы довести до конца:»' in bare
