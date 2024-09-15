"""
Author: vic123 zhangzc_efz@163.com
Date: 2024-09-03 16:48:14
LastEditors: vic123 zhangzc_efz@163.com
LastEditTime: 2024-09-12 18:13:16
FilePath: \Video-Streaming-Using-WebSockets\server\server.py
Description: 

Copyright (c) 2024 by vic123, All Rights Reserved. 
"""

import json
import websockets
import asyncio
import cv2, base64
import sys
import base64

port = 5000

print("Started server on port:", port)


async def send(websocket):
    print("Client Connected!")
    await websocket.send("Connection Established")
    try:
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            _, frame = cap.read()
            encoded = cv2.imencode(".jpg", frame)[1]
            data = str(base64.b64encode(encoded))
            data = data[2 : len(data) - 1]
            await websocket.send(data)

        cap.release()
    except:
        print("Client Disconnected!")
        cap.release()


async def receive(websocket):
    # try:
    cnt = 0
    while True:
        message = await websocket.recv()  # Wait for a message from the client
        cnt += 1
        if (
            sys.getsizeof(message) < 2**10
        ):  # Todo: tell picture from string message in a more elegant way

            print("Received message from client:", message)
            cnt = 0
        else:
            message = json.loads(message)
            time = message["timestamp1"]
            print("Picture recieved from client at time:", time)
            pic_base64 = message["image"]
            pic = base64.b64decode(pic_base64)
            with open("picture\\pic" + str(time) + ".jpg", "wb") as f:
                f.write(pic)


async def handle(websocket, path):
    send_task = asyncio.create_task(send(websocket))
    receive_task = asyncio.create_task(receive(websocket))
    await asyncio.gather(send_task, receive_task)


async def main():
    async with websockets.serve(handle, "0.0.0.0", 5000, max_size=2**30):
        print("WebSocket服务器已启动，监听端口 5000")
        await asyncio.Future()


asyncio.run(main())
