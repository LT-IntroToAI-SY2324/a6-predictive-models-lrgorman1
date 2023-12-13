# Part 4 - Classification Writeup

After completing `a6_part4.py` answer the following questions

## Questions to answer

1. Comment out the StandardScaler and re-run your test. How accurate is the model? Why is that?
The model is way less accurate because without standard scaler the models info is not arranged in a standard distribution

2. How accurate is the model with the StandardScaler? Is this model accurate enough for the given use case? Explain.
It is fairly accurate almost 80%. I don't think this should be used in this case because it coould potentially hurt finances.

3. Looking at the predicted and actual results, how did the model do? Was there a pattern to the inputs that the model was incorrect about?
I think that the model did fairly well considering its accuracy but I didnt really see a pattern within the models inputs.

4. Would a 34 year old Female who makes 56000 a year buy an SUV according to the model? Remember to scale the data before running it through the model.

She would not buy the SUV