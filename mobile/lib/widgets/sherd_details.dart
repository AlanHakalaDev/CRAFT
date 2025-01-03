import 'dart:io';

import 'package:flutter/material.dart';
import 'package:intl/intl.dart';

class SherdDetails extends StatelessWidget {
  const SherdDetails({
    super.key,
    required this.imageUrl,
    required this.title,
    required this.details,
    required this.timestamp,
    required this.fromHive,
    required this.lattitude,
    required this.longitude,
  });

  final String imageUrl;
  final String title;
  final Map<dynamic, dynamic> details;
  final DateTime timestamp;
  final bool fromHive;
  final double lattitude;
  final double longitude;

  String formatTimestamp(DateTime timestamp) {
    // Convert Timestamp to DateTime
    DateTime dateTime = timestamp;

    // Create a DateFormat object for MM/DD/YY
    DateFormat formatter = DateFormat('MM/dd/yy');

    // Format the DateTime to a string
    return formatter.format(dateTime);
  }

  @override
  Widget build(BuildContext context) {
    return Dialog(
      child: Container(
        decoration: BoxDecoration(
          color: Theme.of(context).colorScheme.surface,
          borderRadius: BorderRadius.circular(8),
          boxShadow: const [
            BoxShadow(
              color: Colors.black12,
              blurRadius: 4,
              offset: Offset(0, 2),
            ),
          ],
        ),
        width: MediaQuery.of(context).size.width / 1.2,
        padding: const EdgeInsets.all(16.0),
        child: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Center(
                child: Text(
                  title,
                  style: const TextStyle(
                    fontSize: 30,
                    fontFamily: 'Uber',
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
              const SizedBox(height: 16),
              ClipRRect(
                borderRadius: BorderRadius.circular(5),
                child: (!fromHive)
                    ? Image.network(
                        imageUrl,
                        width: double.infinity,
                        fit: BoxFit.cover,
                      )
                    : Image.file(
                        File(imageUrl),
                        width: double.infinity,
                        fit: BoxFit.cover,
                      ),
              ),
              ExpansionTile(
                title: const Text('Model Confidence'),
                children: [
                  ...details.entries.map((entry) =>
                      Text('${entry.key}: ${entry.value.toStringAsFixed(3)}')),
                  const SizedBox(height: 16),
                ],
              ),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 16.0),
                child: Text(
                  'Classification Date: ${formatTimestamp(timestamp)}',
                ),
              ),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 16.0),
                child: Text(
                  'Approximate Latitude: ${lattitude.toStringAsFixed(4)}',
                ),
              ),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 16.0),
                child: Text(
                  'Approximate Longitude: ${longitude.toStringAsFixed(4)}',
                ),
              ),
              const SizedBox(height: 16),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  FilledButton(
                    onPressed: () {
                      Navigator.of(context).pop();
                    },
                    child: const Text('Dismiss'),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}

void showSherdDetailsDialog(
  BuildContext context, {
  required String imageUrl,
  required String title,
  required Map<dynamic, dynamic> details,
  required DateTime timestamp,
  required bool fromHive,
  required double lattitude,
  required double longitude,
}) {
  showDialog(
    context: context,
    builder: (BuildContext context) {
      return SherdDetails(
        imageUrl: imageUrl,
        title: title,
        details: details,
        timestamp: timestamp,
        fromHive: fromHive,
        lattitude: lattitude,
        longitude: longitude,
      );
    },
  );
}
