# Midicara - abstract

1. Introduce the topic and what motivated our interest in it
2. Summarize general previous research
3. Identify the methods or approaches used that differ (in our project)
4. Identify the outcomes or conclusions drawn from our project

1. Accessible musical instruments foster inclusiveness and empower the instrumentalist in the realm of musical performance. Accessible instruments are those that can be played by an instrumentalist with motor or physical disability, such as lost dexterity in the fingers or hands, without any compromise in technique or progress. The majority of instruments assume full motor capability, which definitively bars individuals with such disabilities [3, 1].
2. Numerous adaptations of accessible musical instruments exist, though few have emerged into widespread use. Some notable implementations are the EyeHarp, which uses gaze tracking projected onto a pitch wheel that plays arpeggiated notes, the Jamboxx, a mouthpiece for pneumatic input that can be moved by the user's lips, BCMI Piano, which uses EEG signals that map to musical output, and HipDisk, which are controllers mounted to the user's body that activate depending on their pose [3, 4, 6].
3. Here we describe our implementation of a facially deterministic controller that outputs MIDI signals. In this monitor/webcam-based implementation, the instrumentalist uses their nose to control a cursor that modifies a parameter of a MIDI-controllable instrument (e.g. pitch, modulation, timbre). By opening and closing their mouth, they can trigger a 'note on' and 'not off' message, respectively; the velocity of which is determined by how widely their mouth is open. This uses `dlib`, a C++ Library for facial recognition, in conjunction with `OpenCV`, a library for image processing [8, 5, 2].
4. The resulting software instrument, dubbed Midicara, uses a visual grid resembling a violin fretboard, similar to that which is used in MidiGrid, to control the MIDI parameter with the nose cursor [3]. The program has minor latency in response time, reading video data at approximately 29FPS. Although heavily limited by the accuracy of `dlib`'s facial recognition model, the implementation demonstrates how accessible instruments do not require expensive equipment or apparatuses, being strictly software based. As `dlib` and others will continue to enhance their facial recognition models, software instruments such as Midicara will be able to operate at a sufficiently high level to make them a viable instrument for public performance.

# References
* [1] Alves-Pinto, Ana, et al. "The case for musical instrument training in cerebral palsy for neurorehabilitation." Neural plasticity 2016 (2016).
* [2] Bradski, G. "The OpenCV Library." Dr. Dobb's Journal of Software Tools, vol. 4, 2000, pp. N/A.
* [3] Frid, Emma. "Accessible digital musical instruments—a review of musical interfaces in inclusive music practice." Multimodal Technologies and Interaction 3.3 (2019): 57.
* [4] “Hands Free, Breath-Powered Instrument - Music for All.” Jamboxx, 12 Dec. 2019, www.jamboxx.com/. 
* [5] Davis E. King. Dlib-ml: A Machine Learning Toolkit. Journal of Machine Learning Research 10, pp. 1755-1758, 2009.
* [6] Lyons, Michael J., Michael Haehnel, and Nobuji Tetsutani. "The mouthesizer: a facial gesture musical interface." Conference Abstracts, Siggraph 2001. Vol. 230. 2001.
* [7] Krägeloh-Mann, Ingeborg, and Christine Cans. "Cerebral palsy update." Brain and development 31.7 (2009): 537-544.
* [8] Sagonas, Christos, et al. "A semi-automatic methodology for facial landmark annotation." Proceedings of the IEEE conference on computer vision and pattern recognition workshops. 2013.
