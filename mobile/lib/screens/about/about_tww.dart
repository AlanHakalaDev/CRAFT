import 'package:craft/screens/about/about_black_mesa.dart';
import 'package:craft/screens/about/about_dogoszhi.dart';
import 'package:craft/screens/about/about_flagstaff.dart';
import 'package:craft/screens/about/about_kayenta.dart';
import 'package:craft/screens/about/about_knaa.dart';
import 'package:craft/screens/about/about_sosi.dart';
import 'package:craft/screens/about/about_tusayan.dart';
import 'package:flutter/material.dart';
import 'package:page_transition/page_transition.dart';

class AboutTww extends StatelessWidget {
  const AboutTww({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        toolbarHeight: 60,
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
              const FittedBox(
                fit: BoxFit.contain,
                child: Text(
                  'Tusayan White Ware',
                  style: TextStyle(
                    fontFamily: 'Uber',
                    fontSize: 60,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ),
              const SizedBox(height: 8),
              ExpansionTile(
                initiallyExpanded: true,
                title: const Text(
                  'Types',
                  style: TextStyle(
                    fontFamily: 'Uber',
                    fontSize: 25,
                    fontWeight: FontWeight.w700,
                  ),
                ),
                subtitle: const Text(
                  'Classifications of Tusayan White Ware',
                  style: TextStyle(
                    fontFamily: 'Uber',
                    fontSize: 15,
                  ),
                ),
                children: [
                  Wrap(
                    children: [
                      TextButton(
                        onPressed: () {
                          Navigator.push(
                            context,
                            PageTransition(
                              child: const AboutKnaa(),
                              type: PageTransitionType.fade,
                            ),
                          );
                        },
                        child: const Text(
                          "Kana'a",
                          style: TextStyle(
                            fontFamily: 'Uber',
                            fontSize: 20,
                          ),
                        ),
                      ),
                      TextButton(
                        onPressed: () {
                          Navigator.push(
                            context,
                            PageTransition(
                              child: const AboutBlackMesa(),
                              type: PageTransitionType.fade,
                            ),
                          );
                        },
                        child: const Text(
                          "Black Mesa",
                          style: TextStyle(
                            fontFamily: 'Uber',
                            fontSize: 20,
                          ),
                        ),
                      ),
                      TextButton(
                        onPressed: () {
                          Navigator.push(
                            context,
                            PageTransition(
                              child: const AboutSosi(),
                              type: PageTransitionType.fade,
                            ),
                          );
                        },
                        child: const Text(
                          "Sosi",
                          style: TextStyle(
                            fontFamily: 'Uber',
                            fontSize: 20,
                          ),
                        ),
                      ),
                      TextButton(
                        onPressed: () {
                          Navigator.push(
                            context,
                            PageTransition(
                              child: const AboutFlagstaff(),
                              type: PageTransitionType.fade,
                            ),
                          );
                        },
                        child: const Text(
                          "Flagstaff",
                          style: TextStyle(
                            fontFamily: 'Uber',
                            fontSize: 20,
                          ),
                        ),
                      ),
                      TextButton(
                        onPressed: () {
                          Navigator.push(
                            context,
                            PageTransition(
                              child: const AboutTusayan(),
                              type: PageTransitionType.fade,
                            ),
                          );
                        },
                        child: const Text(
                          "Tusayan",
                          style: TextStyle(
                            fontFamily: 'Uber',
                            fontSize: 20,
                          ),
                        ),
                      ),
                      TextButton(
                        onPressed: () {
                          Navigator.push(
                            context,
                            PageTransition(
                              child: const AboutDogoszhi(),
                              type: PageTransitionType.fade,
                            ),
                          );
                        },
                        child: const Text(
                          "Dogoszhi",
                          style: TextStyle(
                            fontFamily: 'Uber',
                            fontSize: 20,
                          ),
                        ),
                      ),
                      TextButton(
                        onPressed: () {
                          Navigator.push(
                            context,
                            PageTransition(
                              child: const AboutKayenta(),
                              type: PageTransitionType.fade,
                            ),
                          );
                        },
                        child: const Text(
                          "Kayenta",
                          style: TextStyle(
                            fontFamily: 'Uber',
                            fontSize: 20,
                          ),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 16),
                ],
              ),
              const SizedBox(height: 8),
              const Text(
                'Manufactured',
                style: TextStyle(
                  fontFamily: 'Uber',
                  fontSize: 25,
                  fontWeight: FontWeight.w700,
                ),
              ),
              const Text(
                "Primarily between modern day Tuba City and Kayenta (Kayenta Series).",
                style: TextStyle(
                  fontFamily: 'Uber',
                  fontSize: 18,
                ),
              ),
              const SizedBox(height: 16),
              const Text(
                'Distribution:',
                style: TextStyle(
                  fontFamily: 'Uber',
                  fontSize: 25,
                  fontWeight: FontWeight.w700,
                ),
              ),
              const Text(
                "Throughout much of the US Southwest.",
                style: TextStyle(
                  fontFamily: 'Uber',
                  fontSize: 18,
                ),
              ),
              const SizedBox(height: 16),
              const Text(
                'Physical Characteristics:',
                style: TextStyle(
                  fontFamily: 'Uber',
                  fontSize: 25,
                  fontWeight: FontWeight.w700,
                ),
              ),
              const Text(
                "Paste color: Light gray to white, often with carbon streak.",
                style: TextStyle(
                  fontFamily: 'Uber',
                  fontSize: 18,
                ),
              ),
              const Text(
                "Temper: Mostly fine sand; later types may have sherd or volcanic temper.",
                style: TextStyle(
                  fontFamily: 'Uber',
                  fontSize: 18,
                ),
              ),
              const Text(
                "Surface Treatment: Polished. Slip uncommon in early types, but does occur in later types. Temper does not protrude through surface.",
                style: TextStyle(
                  fontFamily: 'Uber',
                  fontSize: 18,
                ),
              ),
              const Text(
                "Paint: Organic, soaks into surface. Blurred edges. Bidahochi type has mineral paint, sharp edges, dark.",
                style: TextStyle(
                  fontFamily: 'Uber',
                  fontSize: 18,
                ),
              ),
              const SizedBox(height: 16),
              const Text(
                'Other White Ware Types In Area',
                style: TextStyle(
                  fontFamily: 'Uber',
                  fontSize: 25,
                  fontWeight: FontWeight.w700,
                ),
              ),
              const Text(
                "Cibola: Light gray to white paste, often with carbon streak; sand, sherd or both temper; no slip early, thin later; mineral paint.",
                style: TextStyle(
                  fontFamily: 'Uber',
                  fontSize: 18,
                ),
              ),
              const Text(
                "Little Colorado White Ware: Darker paste color; thick white slip; sherd temper, sometimes with sand; organic paint.",
                style: TextStyle(
                  fontFamily: 'Uber',
                  fontSize: 18,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
