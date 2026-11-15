import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'app_controller.dart';
import 'fused_screen.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  // Portrait only: keeps the frame geometry the engine reports stable.
  SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);
  runApp(SpatialNavApp(controller: AppController()));
}

class SpatialNavApp extends StatelessWidget {
  const SpatialNavApp({super.key, required this.controller});

  final AppController controller;

  @override
  Widget build(BuildContext context) => MaterialApp(
        title: 'Spatial nav',
        theme: ThemeData(useMaterial3: true, colorSchemeSeed: Colors.indigo),
        home: FusedScreen(controller: controller),
      );
}
