## an flutter app and python server for 3DGS SLAM

The edge side and the server communicate with a websocket(a duplex protocol). IMU, gesture and picture streams are transferred. The server train 3DGS and stream rendered pictures to the edge side.

### project structure

```python
3DGS
├── Readme.md
├── app
│   ├── ...
│   ├── lib
│   │   ├── VideoStream
│   │   │   ├── VideoStreaming.dart         # widget for streaming camera and imu and recieving 3dgs rendering
│   │   │   └── websocket.dart              # websocket client side
│   │   ├── constants
│   │   │   └── constants.dart              # ip configuration
│   │   ├── gesture_detector3d.dart         # gesture process and forward
│   │   ├── main.dart                   # app entry point
│   │   └── styles
│   │       └── styles.dart
│   ├── test
│   │   ├── client.dart
│   │   ├── gyro_exp.dart
│   │   └── widget_test.dart
│   └── ...
├── server
│   ├── data                            # store imu and camera data from edge app
│   └── pose2img                        # realtime 3dgs rendering(and training(to be implemented))
│       ├── docs_env                        # detailed document for problems met during setting environment
│       ├── render.py                       # render interface
│       ├── server.py                       # websocket server side
│       ├── submodules
│       └── wandb
└── 日志.md                             # record of the project
```

### enviroment set up

#### app

You only need flutter environment to source the app to your mobile phone for the first time. Flutter environment is not required to run the app.

follow through [set up flutter environment](https://book.flutterchina.club/chapter1/install_flutter.html#_1-3-1-%E5%AE%89%E8%A3%85flutter) from `1.3.1` to `1.3.3`. I personally used `vscode` as IDE.

#### server

Follow through `/server/docs_env/Readme.md`. It is recommended to use linux. Refer to `/server/docs_env/fix.md` when installing on windows.

### To Deploy

1. make sure the mobile phone and the laptop are under the same network
2. change the ip address in `app/lib/constants/constants.dart` to laptop's LAN ip
3. start the python server at `server/pose2img/server.py`
4. run app/lib/main.dart on the mobile phone or download the apk(currently not an option)
5. click the connect button on the mobile phone
6. drag and scale the 3dgs rendering with your finger
