import 'package:flutter/material.dart';
import 'dart:async';
import 'dart:convert';

class Drag extends StatefulWidget {
  final Widget child;

  final void Function(dynamic) sendMessage;
  Drag(this.child, this.sendMessage);

  @override
  DragState createState() => DragState();
}

class DragState extends State<Drag> with SingleTickerProviderStateMixin {
  double _yaw = 0.0;
  double _pitch = 0.0;
  double _scale = 1.0;
  Timer? viewTimer;
  @override
  void initState() {
    viewTimer ??= Timer.periodic(Duration(milliseconds: 100), (timer) {
      widget.sendMessage(
          jsonEncode({'yaw': _yaw, 'pitch': _pitch, 'scale': _scale}));
      _scale = 1.0;
    });
    super.initState();
  }

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: <Widget>[
        Positioned(
          child: GestureDetector(
            child: widget.child,
            onScaleUpdate: (ScaleUpdateDetails details) {
              setState(() {
                _yaw += details.focalPointDelta.dx;
                _pitch += details.focalPointDelta.dy;
                _scale = details.scale;
              });
            },
          ),
        )
      ],
    );
  }
}
