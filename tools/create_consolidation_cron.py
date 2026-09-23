"""Create the Agent Server cron for the `consolidator` graph (run manually).

Official API: https://docs.langchain.com/langsmith/cron-jobs (client.crons.create,
schedule in UTC, one new thread per run). Every run makes paid Luna calls,
bounded in deep_agent.consolidator (thread/char caps, ModelCallLimitMiddleware).
Check emergency stop and budget before creating; delete the cron when unused.

    uv run python tools/create_consolidation_cron.py --url https://<deployment> [--schedule "0 */6 * * *"]

The API key is read by langgraph_sdk from LANGGRAPH_API_KEY / LANGSMITH_API_KEY;
it is never printed. Not executed by tests or the release gate.
"""
import argparse
import asyncio
import json
import os

from langgraph_sdk import get_client

ASSISTANT_ID = 'consolidator'
DEFAULT_SCHEDULE = '0 */6 * * *'
METADATA = {'purpose': 'candidate-memory-consolidation'}
INPUT = {'messages': [{'role': 'user', 'content': 'Consolidate recent Brain threads into candidate memory.'}]}


async def create(url: str, schedule: str) -> dict:
    client = get_client(url=url)
    existing = await client.crons.search(metadata=METADATA)
    if existing:
        return {'status': 'exists', 'cron_ids': [item['cron_id'] for item in existing]}
    cron = await client.crons.create(ASSISTANT_ID, schedule=schedule, input=INPUT,
                                     on_run_completed='delete',
                                     metadata=METADATA)
    return {'status': 'created', 'cron_id': cron.get('cron_id'), 'schedule': schedule}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--url', default=os.environ.get('LANGGRAPH_DEPLOYMENT_URL'))
    parser.add_argument('--schedule', default=DEFAULT_SCHEDULE)
    args = parser.parse_args()
    if not args.url:
        parser.error('--url or LANGGRAPH_DEPLOYMENT_URL is required')
    print(json.dumps(asyncio.run(create(args.url, args.schedule)), ensure_ascii=False))


if __name__ == '__main__':
    main()
