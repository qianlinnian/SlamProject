/*
 * @Author: vic123 zhangzc_efz@163.com
 * @Date: 2024-09-03 16:48:14
 * @LastEditors: vic123 zhangzc_efz@163.com
 * @LastEditTime: 2024-09-12 19:25:31
 * @FilePath: \app\lib\VideoStream\VideoStreaming.dart
 * @Description: 
 * 
 * Copyright (c) 2024 by vic123, All Rights Reserved. 
 */
import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:app/VideoStream/websocket.dart';
import 'package:app/constants/constants.dart';
import 'package:flutter/material.dart';
import 'package:app/styles/styles.dart';
import 'package:sensors_plus/sensors_plus.dart';
import 'package:camera/camera.dart';
import 'package:app/gesture_detector3d.dart';

late List<CameraDescription> _cameras;

class VideoStream extends StatefulWidget {
  const VideoStream({Key? key}) : super(key: key);

  @override
  State<VideoStream> createState() => _VideoStreamState();
}

class _VideoStreamState extends State<VideoStream> {
  final WebSocket _socket = WebSocket(Constants.videoWebsocketURL);
  bool _isConnected = false;
  Timer? picTimer;
  late CameraController controller;
  @override
  void initState() {
    _loadCam();
    accelerometerEventStream(samplingPeriod: SensorInterval.uiInterval).listen(
      (AccelerometerEvent event) {
        if (_isConnected) {
          _socket.sendMessage(jsonEncode({
            'x': event.x,
            'y': event.y,
            'z': event.z,
            'timestamp': event.timestamp.microsecondsSinceEpoch,
            'code': "accelerometer"
          }));
        }
      },
      onError: (error) {
        // Logic to handle error
        // Needed for Android in case sensor is not available
      },
      cancelOnError: true,
    );
    gyroscopeEventStream(samplingPeriod: SensorInterval.uiInterval).listen(
      (GyroscopeEvent event) {
        if (_isConnected) {
          _socket.sendMessage(jsonEncode({
            'x': event.x,
            'y': event.y,
            'z': event.z,
            'timestamp': event.timestamp.microsecondsSinceEpoch,
            'code': "gyroscope"
          }));
        }
      },
      onError: (error) {
        // Logic to handle error
        // Needed for Android in case sensor is not available
      },
      cancelOnError: true,
    );
    startPictureTimer();
    super.initState();
  }

  Future<void> _loadCam() async {
    _cameras = await availableCameras();
    controller = CameraController(_cameras[0], ResolutionPreset.medium);

    controller.initialize().then((_) {
      if (!mounted) {
        return;
      }
      setState(() {});
    }).catchError((Object e) {
      if (e is CameraException) {
        switch (e.code) {
          case 'CameraAccessDenied':
            print(e.code);
            break;
          default:
            print(e.code);
            break;
        }
      }
    });
  }

  void connect(BuildContext context) async {
    _socket.connect();
    setState(() {
      _isConnected = true;
    });
  }

  void disconnect() {
    setState(() {
      _isConnected = false;
    });
    _socket.disconnect();
  }

  void startPictureTimer() {
    if (picTimer != null && picTimer!.isActive) {
      return;
    }
    picTimer = Timer.periodic(const Duration(milliseconds: 500), (Timer timer) {
      if (_isConnected) {
        sendPicture();
      } else {
        // timer.cancel();
      }
    });
  }

  void sendPicture() async {
    int timestamp1 = DateTime.now().millisecondsSinceEpoch;
    final XFile image = await controller.takePicture();
    int timestamp2 = DateTime.now().millisecondsSinceEpoch;
    Uint8List imageBytes = await image.readAsBytes();

    Map<String, dynamic> message = {
      'timestamp1': timestamp1,
      'timestamp2': timestamp2,
      'image': base64Encode(imageBytes),
    };
    _socket.sendMessage(jsonEncode(message));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("3DGS RealTime Rendering"),
      ),
      body: Padding(
        padding: const EdgeInsets.all(20.0),
        child: Center(
          child: Column(
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  ElevatedButton(
                    onPressed: () => connect(context),
                    style: Styles.buttonStyle,
                    child: const Text("Connect"),
                  ),
                  ElevatedButton(
                    onPressed: disconnect,
                    style: Styles.buttonStyle,
                    child: const Text("Disconnect"),
                  ),
                ],
              ),
              const SizedBox(
                height: 50.0,
              ),
              _isConnected
                  ? Drag(
                      StreamBuilder(
                        stream: _socket.stream,
                        builder: (context, snapshot) {
                          if (!snapshot.hasData) {
                            return const CircularProgressIndicator();
                          }
                          if (snapshot.connectionState ==
                              ConnectionState.done) {
                            return const Center(
                              child: Text("Connection Closed !"),
                            );
                          }
                          if (snapshot.data == "Connection Established") {
                            return const Center(
                              child: Text("Connection Closed !"),
                            );
                          }
                          //? Working for single frames
                          return Image.memory(
                            Uint8List.fromList(
                              base64Decode(
                                (snapshot.data.toString()),
                              ),
                            ),
                            width: 1000,
                            gaplessPlayback: true,
                            excludeFromSemantics: true,
                          );
                        },
                      ),
                      _socket.sendMessage)
                  : const Text("Initiate Connection")
            ],
          ),
        ),
      ),
    );
  }
}
