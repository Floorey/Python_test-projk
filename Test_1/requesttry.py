import requests
from requests import Response
import asyncio
from asyncio import Task

async def fetch_status(url: str) -> dict:
    print(f'Fetching status for: {url}')
    responce: Response = await asyncio.to_thread(requests.get, url, None)
    print('Done')
    return {'status': responce.status_code, 'url': url}


async def main() -> None:
    apple_task: Task[dict] = asyncio.create_task(fetch_status('https://www.apple.com/de/'))
    google_task: Task[dict] = asyncio.create_task(fetch_status('https://www.google.com/'))
    amazon_task: Task[dict] = asyncio.create_task(fetch_status('https://www.amazon.de/'))
    
    
    
    apple_status: dict = await apple_task
    google_status: dict = await google_task
    amazon_task: dict = await amazon_task

    print(apple_status)
    print(google_status)



if __name__ == '__main__':
    asyncio.run(main=main())
