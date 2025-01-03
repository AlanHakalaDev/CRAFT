import 'package:flutter/material.dart';

class AboutCraft extends StatelessWidget {
  const AboutCraft({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        toolbarHeight: 80,
        title: const Padding(
          padding: EdgeInsets.symmetric(horizontal: 8.0),
          child: Text(
            'About',
            style: TextStyle(
                fontFamily: 'Uber', fontSize: 60, fontWeight: FontWeight.w700),
          ),
        ),
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const SizedBox(height: 16),
              const Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    "The goal of CRAFT is to increase the consistency of classification streamline the process of classifying Tusayan White Ware pottery shards by archeologists, use use cutting edge ConvNeXT model for imagge classification to ensure consistency and accuracy for TWW classifications.",
                    style: TextStyle(
                      fontFamily: 'Uber',
                      fontSize: 18,
                    ),
                  ),
                  SizedBox(height: 16),
                  Text(
                    'Project Sponsor',
                    style: TextStyle(
                      fontFamily: 'Uber',
                      fontSize: 25,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  SizedBox(height: 8),
                  Text(
                    "    Dr. Leszek Pawlowicz ",
                    style: TextStyle(
                      fontFamily: 'Uber',
                      fontSize: 18,
                    ),
                  ),
                  Text(
                    "    Leszek.Pawlowicz@nau.edu",
                    style: TextStyle(
                      fontFamily: 'Uber',
                      fontSize: 15,
                    ),
                  ),
                  SizedBox(
                    height: 16,
                  ),
                  Text(
                    'Project Developers',
                    style: TextStyle(
                      fontFamily: 'Uber',
                      fontSize: 25,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  SizedBox(height: 8),
                  Text(
                    "    Kimberly Allison",
                    style: TextStyle(
                      fontFamily: 'Uber',
                      fontSize: 18,
                    ),
                  ),
                  Text(
                    "    Aadarsha Bastola",
                    style: TextStyle(
                      fontFamily: 'Uber',
                      fontSize: 18,
                    ),
                  ),
                  Text(
                    "    Alan Hakala",
                    style: TextStyle(
                      fontFamily: 'Uber',
                      fontSize: 18,
                    ),
                  ),
                  Text(
                    "    Nick Wiley",
                    style: TextStyle(
                      fontFamily: 'Uber',
                      fontSize: 18,
                    ),
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
