import asyncio

class EventBus:
    def __init__(self):
        self.q = asyncio.Queue()

    async def publish(self, event):
        await self.q.put(event)

    async def subscribe(self):
        while True:
            e = await self.q.get()
            yield e
