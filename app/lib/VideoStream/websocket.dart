/*
 * @Author: vic123 zhangzc_efz@163.com
 * @Date: 2024-09-03 16:48:14
 * @LastEditors: vic123 zhangzc_efz@163.com
 * @LastEditTime: 2024-09-09 18:13:10
 * @FilePath: \app\lib\VideoStream\websocket.dart
 * @Description: 
 * 
 * Copyright (c) 2024 by vic123, All Rights Reserved. 
 */
import 'dart:async';
import 'package:app/constants/constants.dart';
import 'package:web_socket_channel/web_socket_channel.dart';
import 'package:web_socket_channel/status.dart' as status;

class WebSocket {
  // ------------------------- Members ------------------------- //
  late String url;
  WebSocketChannel? _channel;
  // StreamController<bool> streamController = StreamController<bool>.broadcast();

  // ---------------------- Getter Setters --------------------- //
  String get getUrl {
    return url;
  }

  set setUrl(String url) {
    this.url = url;
  }

  Stream<dynamic> get stream {
    if (_channel != null) {
      return _channel!.stream;
    } else {
      throw WebSocketChannelException("The connection was not established !");
    }
  }

  // --------------------- Constructor ---------------------- //
  WebSocket(this.url);

  // ---------------------- Functions ----------------------- //

  /// Connects the current application to a websocket
  void connect() async {
    _channel = WebSocketChannel.connect(Uri.parse(Constants.videoWebsocketURL));
  }

//
  void sendMessage(dynamic message) {
    if (_channel != null && _channel!.closeCode == null) {
      try {
        _channel!.sink.add(message);
      } catch (e) {
        print(e);
      }
    }
  }

  /// Disconnects the current application from a websocket
  void disconnect() {
    if (_channel != null) {
      _channel!.sink.close(status.goingAway);
    }
  }
}
