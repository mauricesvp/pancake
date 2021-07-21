import asyncio
import time

import aiofiles
from aiohttp import BaseConnector, ClientSession, ClientConnectorError


async def fetch_html(url: str, session: ClientSession, **kwargs) -> tuple:
    try:
        r = await session.request(method="GET", url=url, **kwargs)
        side = url[42]
        f = await aiofiles.open(
            f"test/1{side}/" + str(time.time())[:13] + ".jpg", mode="wb"
        )
        await f.write(await r.content.read())
        await f.close()
    except:
        pass


async def make_requests(urls: set, **kwargs) -> None:
    async with ClientSession() as session:
        tasks = [
            asyncio.create_task(fetch_html(url=url, session=session, **kwargs))
            for url in urls
        ]

        results = await asyncio.gather(*tasks)


if __name__ == "__main__":
    URL_PREFIX = "https://media.dcaiti.tu-berlin.de/tccams/"
    URL_SUFFIX = "/jpg/image.jpg?camera=1&compression=50"
    l = URL_PREFIX + "1l" + URL_SUFFIX
    c = URL_PREFIX + "1c" + URL_SUFFIX
    r = URL_PREFIX + "1r" + URL_SUFFIX
    urls = [l, c, r] * 500

    print(time.time())
    asyncio.run(make_requests(urls=urls))
    print(time.time())
