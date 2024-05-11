# ScorEasy
A web application built using Flask utilizing YOLOv8 objection detection algorithm to automate the grading process.

**Features:**

Supports multiple choice questions

User friendly interface

Eliminates human error and reduces grading time by 95%

**Working:**

The web interface takes the answer sheet as an input. Each question from the answer sheet is passed to the model which accurately predicts the user marked option/s using labels that include
'a_selected', 'b_selected', 'c_selected' or 'd_selected'.

**Constraints:**

Currently limited to four choices (i.e. A, B, C and D) and test taker is expected to circle the correct option

Dataset is limited as it was created from scratch and lack of volunteers

**Future work:**

The idea is to ultimately turn this into a mobile app where user can simply scan and see the final grades

Support for variety of markings (Ex: if a test taker prefers check mark on correct answer instead of circling)

Add more samples to the dataset
