## an flutter app and python server for 3DGS SLAM (IN PROGRESS)

### project structure

### function

the mobile app and the server are connected using websocket.

the mobile app will transfer IMU data and picture from camera and a view(from finger gesture on the screen) to the server

the server stream a video to the mobile app

### To Deploy

1. make sure the mobile phone and the laptop are under the same network
2. change the ip address in app/lib/constants/constants.dart to laptop's LAN ip
3. start the python server
4. run app/lib/main.dart on the mobile phone(must have flutter/dart environment) or download the apk(currently not an option, it's still a work in progress)
5. click the connect button on the mobile phone
