import 'dart:io';

import 'package:craft/screens/edit_results.dart';
import 'package:flutter/material.dart';
import 'package:geolocator/geolocator.dart';
import 'package:image_cropper/image_cropper.dart';
import 'package:image_picker/image_picker.dart';
import 'package:page_transition/page_transition.dart';

class HomePage extends StatefulWidget {
  static const String id = 'home';
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final ImagePicker picker = ImagePicker();
  File? selectedImage;
  String? classificaitonData;
  Position? currentPosition;

  Future pickAndCropImage(ImageSource source) async {
    final pickedImage =
        await picker.pickImage(source: source, imageQuality: 50);

    CroppedFile? croppedFile = await ImageCropper().cropImage(
      sourcePath: pickedImage!.path,
      aspectRatio: const CropAspectRatio(ratioX: 1, ratioY: 1),
      uiSettings: [
        AndroidUiSettings(
            initAspectRatio: CropAspectRatioPreset.original,
            lockAspectRatio: true),
        IOSUiSettings(
          aspectRatioLockEnabled: true,
          title: 'Crop Image',
        ),
      ],
    );

    setState(() {
      selectedImage = File(croppedFile!.path);
    });
  }

  void resetScreen() {
    setState(() {
      selectedImage = null;
      classificaitonData = null;
      currentPosition = null;
    });
  }

  void classifyImage() async {
    Position pos = await _determinePosition();
    setState(() {
      currentPosition = pos;
      classificaitonData =
          "Flagstaff: Confidence 0.123\nBlack Mesa: Confidence 0.123\nKnaa: Confidence 0.123";
    });
  }

  Future<Position> _determinePosition() async {
    bool serviceEnabled;
    LocationPermission permission;

    // Test if location services are enabled.
    serviceEnabled = await Geolocator.isLocationServiceEnabled();
    if (!serviceEnabled) {
      // Location services are not enabled don't continue
      // accessing the position and request users of the
      // App to enable the location services.
      return Future.error('Location services are disabled.');
    }

    permission = await Geolocator.checkPermission();
    if (permission == LocationPermission.denied) {
      permission = await Geolocator.requestPermission();
      if (permission == LocationPermission.denied) {
        // Permissions are denied, next time you could try
        // requesting permissions again (this is also where
        // Android's shouldShowRequestPermissionRationale
        // returned true. According to Android guidelines
        // your App should show an explanatory UI now.
        return Future.error('Location permissions are denied');
      }
    }

    if (permission == LocationPermission.deniedForever) {
      // Permissions are denied forever, handle appropriately.
      return Future.error(
          'Location permissions are permanently denied, we cannot request permissions.');
    }

    // When we reach here, permissions are granted and we can
    // continue accessing the position of the device.
    return await Geolocator.getCurrentPosition(
        desiredAccuracy: LocationAccuracy.low);

    // TODO: Randomize the geoloation by a little bit
  }

  void editClassification() {
    Navigator.push(context,
        PageTransition(child: EditResults(), type: PageTransitionType.fade));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            const SizedBox(
              height: 16,
            ),
            const Padding(
              padding: EdgeInsets.symmetric(vertical: 16),
              child: SizedBox(
                width: double.infinity,
                child: FittedBox(
                  fit: BoxFit.contain,
                  child: Text(
                    'Tuyscan White\nWare Classification',
                    style: TextStyle(
                      fontFamily: 'Uber',
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ),
              ),
            ),
            selectedImage == null
                ? Column(
                    children: [
                      const Padding(
                        padding: EdgeInsets.all(16.0),
                        child: Text('Load Image to Get Started'),
                      ),
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                        children: [
                          GestureDetector(
                            onTap: () => pickAndCropImage(ImageSource.camera),
                            child: Container(
                              decoration: BoxDecoration(
                                  color: Theme.of(context)
                                      .colorScheme
                                      .primary, // select color from current theme scheme
                                  borderRadius: const BorderRadius.all(
                                      Radius.circular(5))),
                              child: Padding(
                                padding: const EdgeInsets.symmetric(
                                    horizontal: 40, vertical: 8),
                                child: Column(
                                  children: [
                                    Icon(
                                      Icons.camera_alt_rounded,
                                      size: 50,
                                      color: Theme.of(context)
                                          .colorScheme
                                          .onPrimary,
                                    ),
                                    Text(
                                      'Camera',
                                      style: TextStyle(
                                        color: Theme.of(context)
                                            .colorScheme
                                            .onPrimary,
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                            ),
                          ),
                          GestureDetector(
                            onTap: () => pickAndCropImage(ImageSource.gallery),
                            child: Container(
                              decoration: BoxDecoration(
                                  color: Theme.of(context)
                                      .colorScheme
                                      .primary, // select color from current theme scheme
                                  borderRadius: const BorderRadius.all(
                                      Radius.circular(5))),
                              child: Padding(
                                padding: const EdgeInsets.symmetric(
                                    horizontal: 40, vertical: 8),
                                child: Column(
                                  children: [
                                    Icon(
                                      Icons.image_rounded,
                                      size: 50,
                                      color: Theme.of(context)
                                          .colorScheme
                                          .onPrimary,
                                    ),
                                    Text(
                                      'Gallery',
                                      style: TextStyle(
                                        color: Theme.of(context)
                                            .colorScheme
                                            .onPrimary,
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                            ),
                          ),
                        ],
                      ),
                    ],
                  )
                : Container(),
            selectedImage != null
                ? AspectRatio(
                    aspectRatio: 1,
                    child: Container(
                      decoration: BoxDecoration(
                          color: Theme.of(context)
                              .colorScheme
                              .secondaryContainer, // select color from current theme scheme
                          borderRadius:
                              const BorderRadius.all(Radius.circular(5))),
                      width: MediaQuery.of(context).size.width,
                      // height: 100,
                      child: Center(
                          child: selectedImage != null
                              ? ClipRRect(
                                  borderRadius: BorderRadius.circular(5),
                                  child: Image.file(
                                    selectedImage!,
                                    fit: BoxFit.cover,
                                    width: MediaQuery.of(context).size.width,
                                  ))
                              : Container()),
                    ),
                  )
                : Container(),
            const SizedBox(
              height: 30,
            ),
            classificaitonData != null
                ? Container(
                    decoration: BoxDecoration(
                        color: Theme.of(context)
                            .colorScheme
                            .primaryContainer, // select color from current theme scheme
                        borderRadius:
                            const BorderRadius.all(Radius.circular(5))),
                    width: MediaQuery.of(context).size.width,
                    child: Padding(
                      padding: const EdgeInsets.all(12.0),
                      child: Text(currentPosition.toString()),
                    ),
                  )
                : Container(),
            const SizedBox(
              height: 16,
            ),
            selectedImage != null && classificaitonData == null
                ? FilledButton(
                    onPressed: classifyImage, child: const Text('Classify'))
                : Container(),
            selectedImage != null && classificaitonData == null
                ? TextButton(
                    onPressed: resetScreen, child: const Text('Clear Image'))
                : Container(),
            selectedImage != null && classificaitonData != null
                ? FilledButton(
                    onPressed: classifyImage,
                    child: const Text('Save Classification'))
                : Container(),
            selectedImage != null && classificaitonData != null
                ? TextButton(
                    onPressed: resetScreen,
                    child: const Text('Clear Image and Classification'))
                : Container(),
            selectedImage != null && classificaitonData != null
                ? TextButton(
                    onPressed: editClassification,
                    child: const Text('Edit Classification'))
                : Container(),
          ],
        ),
      ),
    );
  }
}
