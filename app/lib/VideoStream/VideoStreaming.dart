/*
 * @Author: vic123 zhangzc_efz@163.com
 * @Date: 2024-09-03 16:48:14
 * @LastEditors: vic123 zhangzc_efz@163.com
 * @FilePath: \app\lib\VideoStream\VideoStreaming.dart
 */
import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:app/VideoStream/websocket.dart';
import 'package:app/constants/constants.dart';
import 'package:app/gesture_detector3d.dart';
import 'package:app/styles/styles.dart';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import 'package:sensors_plus/sensors_plus.dart';

late List<CameraDescription> _cameras;

class VideoStream extends StatefulWidget {
  const VideoStream({Key? key}) : super(key: key);

  @override
  State<VideoStream> createState() => _VideoStreamState();
}

class _VideoStreamState extends State<VideoStream> {
  final WebSocket _socket = WebSocket(Constants.videoWebsocketURL);
  final int _minFrameIntervalMs = 300;
  bool _isConnected = false;
  bool _isConnecting = false;
  bool _isCapturing = false;
  bool _streamStarted = false;
  bool _loggedFrameSize = false;
  int _lastSentAtMs = 0;
  String _frameSizeLabel = "Capture size: detecting...";
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
      onError: (error) {},
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
      onError: (error) {},
      cancelOnError: true,
    );
    super.initState();
  }

  Future<void> _loadCam() async {
    _cameras = await availableCameras();
    controller = CameraController(
      _cameras[0],
      ResolutionPreset.low,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );

    controller.initialize().then((_) async {
      if (!mounted) {
        return;
      }
      await _startCameraStream();
      setState(() {});
    }).catchError((Object e) {
      if (e is CameraException) {
        print(e.code);
      }
    });
  }

  Future<void> _startCameraStream() async {
    if (_streamStarted || !controller.value.isInitialized) {
      return;
    }
    await controller.startImageStream((CameraImage cameraImage) {
      if (!_loggedFrameSize) {
        print("CAMERA_FRAME_SIZE: ${cameraImage.width}x${cameraImage.height}");
        if (mounted) {
          setState(() {
            _frameSizeLabel =
                "Capture size: ${cameraImage.width} x ${cameraImage.height}";
          });
        }
        _loggedFrameSize = true;
      }
      final int now = DateTime.now().millisecondsSinceEpoch;
      if (!_isConnected || _isCapturing) {
        return;
      }
      if (now - _lastSentAtMs < _minFrameIntervalMs) {
        return;
      }
      _lastSentAtMs = now;
      _sendCameraImage(cameraImage, now);
    });
    _streamStarted = true;
  }

  Future<void> connect(BuildContext context) async {
    if (_isConnected || _isConnecting) {
      return;
    }
    setState(() {
      _isConnecting = true;
    });
    try {
      await _socket.connect();
      if (!mounted) {
        return;
      }
      setState(() {
        _isConnected = true;
      });
    } finally {
      if (mounted) {
        setState(() {
          _isConnecting = false;
        });
      }
    }
  }

  void disconnect() {
    setState(() {
      _isConnected = false;
      _isConnecting = false;
    });
    _socket.disconnect();
  }

  Future<void> _sendCameraImage(CameraImage cameraImage, int timestamp1) async {
    if (_isCapturing) return;
    _isCapturing = true;
    try {
      final Stopwatch stopwatch = Stopwatch()..start();
      final Uint8List imageBytes = await _encodeCameraImage(cameraImage);
      stopwatch.stop();
      final int timestamp2 = DateTime.now().millisecondsSinceEpoch;
      print("FRAME_ENCODE: ${stopwatch.elapsedMilliseconds}ms");

      final Map<String, dynamic> message = {
        'timestamp1': timestamp1,
        'timestamp2': timestamp2,
        'image': base64Encode(imageBytes),
      };
      _socket.sendMessage(jsonEncode(message));
    } finally {
      _isCapturing = false;
    }
  }

  Future<Uint8List> _encodeCameraImage(CameraImage cameraImage) async {
    final img.Image rgbImage = _cameraImageToRgb(cameraImage);
    return Uint8List.fromList(img.encodeJpg(rgbImage, quality: 70));
  }

  img.Image _cameraImageToRgb(CameraImage cameraImage) {
    switch (cameraImage.format.group) {
      case ImageFormatGroup.yuv420:
        return _convertYuv420(cameraImage);
      case ImageFormatGroup.bgra8888:
        return _convertBgra8888(cameraImage);
      default:
        throw UnsupportedError(
          'Unsupported image format: ${cameraImage.format.group}',
        );
    }
  }

  img.Image _convertBgra8888(CameraImage cameraImage) {
    final int width = cameraImage.width;
    final int height = cameraImage.height;
    final Uint8List bytes = cameraImage.planes[0].bytes;
    final img.Image image = img.Image(width, height);

    int offset = 0;
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final int b = bytes[offset];
        final int g = bytes[offset + 1];
        final int r = bytes[offset + 2];
        image.setPixelRgba(x, y, r, g, b);
        offset += 4;
      }
    }
    return image;
  }

  img.Image _convertYuv420(CameraImage cameraImage) {
    final int width = cameraImage.width;
    final int height = cameraImage.height;
    final img.Image image = img.Image(width, height);

    final Plane planeY = cameraImage.planes[0];
    final Plane planeU = cameraImage.planes[1];
    final Plane planeV = cameraImage.planes[2];

    for (int y = 0; y < height; y++) {
      final int uvRow = y ~/ 2;
      for (int x = 0; x < width; x++) {
        final int uvCol = x ~/ 2;
        final int yIndex = y * planeY.bytesPerRow + x;
        final int uvIndex = uvRow * planeU.bytesPerRow + uvCol * planeU.bytesPerPixel!;

        final int yp = planeY.bytes[yIndex];
        final int up = planeU.bytes[uvIndex];
        final int vp = planeV.bytes[uvIndex];

        final int r = (yp + 1.402 * (vp - 128)).round().clamp(0, 255);
        final int g = (yp - 0.344136 * (up - 128) - 0.714136 * (vp - 128))
            .round()
            .clamp(0, 255);
        final int b = (yp + 1.772 * (up - 128)).round().clamp(0, 255);

        image.setPixelRgba(x, y, r, g, b);
      }
    }
    return image;
  }

  @override
  void dispose() {
    if (_streamStarted && controller.value.isStreamingImages) {
      controller.stopImageStream();
    }
    _socket.disconnect();
    controller.dispose();
    super.dispose();
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
                    onPressed: (_isConnected || _isConnecting)
                        ? null
                        : () => connect(context),
                    style: Styles.buttonStyle,
                    child: Text(_isConnecting ? "Connecting..." : "Connect"),
                  ),
                  ElevatedButton(
                    onPressed: disconnect,
                    style: Styles.buttonStyle,
                    child: const Text("Disconnect"),
                  ),
                ],
              ),
              const SizedBox(height: 10.0),
              Text("Target: ${Constants.videoWebsocketURL}",
                  style: const TextStyle(fontSize: 14, color: Colors.grey)),
              const SizedBox(height: 8.0),
              Text(
                _frameSizeLabel,
                style: const TextStyle(fontSize: 13, color: Colors.grey),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 8.0),
              const Text(
                "Display guide: row 1 = main rendered map; row 2 = auxiliary view panels.",
                style: TextStyle(fontSize: 13, color: Colors.grey),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 20.0),
              _isConnected
                  ? Drag(
                      StreamBuilder(
                        stream: _socket.stream,
                        builder: (context, snapshot) {
                          if (!snapshot.hasData) {
                            return const CircularProgressIndicator();
                          }
                          if (snapshot.connectionState == ConnectionState.done) {
                            return const Center(
                              child: Text("Connection Closed !"),
                            );
                          }
                          if (snapshot.data == "Connection Established") {
                            return const Center(
                              child: Text("Connection Closed !"),
                            );
                          }
                          return Image.memory(
                            Uint8List.fromList(
                              base64Decode(snapshot.data.toString()),
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
