import asyncio
import base64
import csv
import json
import os
import shutil
import sys
import time

import cv2
import numpy as np
import websockets


class WebsocketServer:
    def __init__(
        self,
        host="0.0.0.0",
        port=5000,
        save=False,
        server2tracker_queue=None,
        mapper2server_queue=None,
    ):
        self.server2tracker_queue = server2tracker_queue
        self.save = save
        self.host = host
        self.port = port
        self.mapper2server_queue = mapper2server_queue

        # [(timestamp, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z), ...]
        self.accumulated_imu = []

        self.yaw, self.pitch, self.raw, self.x, self.y, self.z, self.offsetx, self.offsety, self.offsetz = (
            1.5,
            0.3,
            3.1415926,
            0.4508,
            0.3173,
            0.3940,
            0.0,
            0.0,
            0.0,
        )
        self.frame = self.draw_corr()
        self.lock = asyncio.Lock()

        if self.save:
            folder_name = str(time.time())
            pic_save_dir = os.path.join("../mobile_data/", folder_name, "pic")
            self.save_dir = os.path.join("../mobile_data/", folder_name)
            if not os.path.exists(pic_save_dir):
                os.mkdir(pic_save_dir)

    def delete_all_contents(self, folder_path):
        if not os.path.isdir(folder_path):
            print(f"Error: {folder_path} is not a valid folder path")
            return

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    print(f"Deleted directory: {file_path}")
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")

    def draw_corr(self):
        canvas = np.zeros((480, 720, 3), dtype="uint8")

        numbers = [
            "yaw" + str(self.yaw),
            "pitch" + str(self.pitch),
            "raw" + str(self.raw),
            "x" + str(self.x + self.offsetx),
            "y" + str(self.y + self.offsety),
            "z" + str(self.z + self.offsetz),
        ]
        positions = [(50, 50), (50, 100), (50, 150), (50, 200), (50, 250), (50, 300)]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (255, 255, 255)
        thickness = 2

        for number, position in zip(numbers, positions):
            cv2.putText(canvas, number, position, font, font_scale, color, thickness)
        return canvas

    async def send(self, websocket):
        print("Client Connected!")
        await websocket.send("Connection Established")
        try:
            while True:
                while self.mapper2server_queue.qsize() != 0:
                    self.frame = self.mapper2server_queue.get()
                encoded = cv2.imencode(".jpg", self.frame)[1]
                data = str(base64.b64encode(encoded))
                data = data[2 : len(data) - 1]
                await websocket.send(data)
                await asyncio.sleep(0.1)
        except Exception:
            print("Client Disconnected!")

    async def receive(self, websocket):
        gyro_flag = 0
        acc_flag = 0
        if self.save:
            file_handle = open(os.path.join(self.save_dir, "imu.csv"), mode="w", newline="")
        gyro_acc = {
            "timestamp": 0,
            "omega_x": 0,
            "omega_y": 0,
            "omega_z": 0,
            "alpha_x": 0,
            "alpha_y": 0,
            "alpha_z": 0,
        }
        if self.save:
            writer = csv.DictWriter(file_handle, fieldnames=gyro_acc.keys())
            writer.writeheader()

        while True:
            message = await websocket.recv()
            if sys.getsizeof(message) < 2**10:
                try:
                    message = json.loads(message)
                    if message["code"] == "orientationUpdate":
                        await self.orientationUpdate(message)
                    elif message["code"] == "scaleUpdate":
                        await self.scaleUpdate(message)
                    elif message["code"] == "scaleEnd":
                        await self.scaleEnd(message)
                    elif message["code"] == "accelerometer":
                        gyro_acc["alpha_x"] = message["x"]
                        gyro_acc["alpha_y"] = message["y"]
                        gyro_acc["alpha_z"] = message["z"]
                        if gyro_flag == 0:
                            gyro_acc["timestamp"] = message["timestamp"] * 1000
                        acc_flag = 1
                    elif message["code"] == "gyroscope":
                        gyro_acc["omega_x"] = message["x"]
                        gyro_acc["omega_y"] = message["y"]
                        gyro_acc["omega_z"] = message["z"]
                        if acc_flag == 0:
                            gyro_acc["timestamp"] = message["timestamp"] * 1000
                        gyro_flag = 1
                except Exception:
                    pass

                if gyro_flag == 1 and acc_flag == 1:
                    self.accumulated_imu.append(
                        [
                            gyro_acc["timestamp"],
                            gyro_acc["alpha_x"],
                            gyro_acc["alpha_y"],
                            gyro_acc["alpha_z"],
                            gyro_acc["omega_x"],
                            gyro_acc["omega_y"],
                            gyro_acc["omega_z"],
                        ]
                    )
                    if self.save:
                        writer.writerow(gyro_acc)
                    gyro_flag = 0
                    acc_flag = 0
                continue

            message = json.loads(message)
            pic = base64.b64decode(message["image"])
            nparr = np.frombuffer(pic, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            timestamp = message.get("timestamp2") or message.get("timestamp1") or int(time.time() * 1000)

            if self.server2tracker_queue:
                current_imu = self.accumulated_imu.copy()
                self.accumulated_imu = []
                packet = {
                    "rgb": img,
                    "timestamp": str(timestamp),
                    "imu": current_imu,
                }

                # Favor low latency: when the queue is full, discard the oldest frame
                # so the tracker sees fresher frames instead of building stale backlog.
                if self.server2tracker_queue.qsize() >= 10:
                    try:
                        self.server2tracker_queue.get_nowait()
                        print("server2tracker_queue full, dropped stale frame")
                    except Exception:
                        print("server2tracker_queue full")

                self.server2tracker_queue.put(packet)
                print("put data to server2tracker_queue")

            if self.save:
                with open(os.path.join(self.save_dir, "pic", str(timestamp) + ".png"), "wb") as f:
                    f.write(pic)

    def YawPitchRaw2RotationMatrix(self, yaw, pitch, raw):
        yaw_matrix = np.array(
            [[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]],
            dtype=np.float32,
        )
        pitch_matrix = np.array(
            [
                [1, 0, 0],
                [0, np.cos(pitch), -np.sin(pitch)],
                [0, np.sin(pitch), np.cos(pitch)],
            ],
            dtype=np.float32,
        )
        raw_matrix = np.array(
            [[np.cos(raw), -np.sin(raw), 0], [np.sin(raw), np.cos(raw), 0], [0, 0, 1]],
            dtype=np.float32,
        )
        return np.dot(np.dot(yaw_matrix, pitch_matrix), raw_matrix)

    async def scaleUpdate(self, message):
        rotation_matrix = self.YawPitchRaw2RotationMatrix(self.yaw, self.pitch, self.raw)
        async with self.lock:
            self.offsetx = rotation_matrix[0][2] * (message["scale"] - 1)
            self.offsety = rotation_matrix[1][2] * (message["scale"] - 1)
            self.offsetz = rotation_matrix[2][2] * (message["scale"] - 1)

    async def orientationUpdate(self, message):
        async with self.lock:
            self.yaw += message["yaw"] / 200
            self.pitch -= message["pitch"] / 200

    async def scaleEnd(self, message):
        async with self.lock:
            self.x += self.offsetx
            self.y += self.offsety
            self.z += self.offsetz
            self.offsetx = 0
            self.offsety = 0
            self.offsetz = 0
            self.scale = 1
        print("yaw", self.yaw, "pitch", self.pitch, "raw", self.raw, "x", self.x, "y", self.y, "z", self.z)

    async def handle(self, websocket):
        send_task = asyncio.create_task(self.send(websocket))
        receive_task = asyncio.create_task(self.receive(websocket))
        await asyncio.gather(send_task, receive_task)

    async def run(self):
        async with websockets.serve(self.handle, self.host, self.port, max_size=2**30):
            print(f"WebSocket server listening on ws://{self.host}:{self.port}")
            await asyncio.Future()


# usage
# server = WebsocketServer()
# asyncio.run(server.run())
