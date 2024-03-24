#set page(columns: 2)
#set text(size:6pt)
#show heading.where(
  level: 3
): it => text(
  size: 11pt,
  weight: "bold",
  it.body
)
#set page(margin: (
  top: 0.5cm,
  bottom: 0.5cm,
  x: 0.5cm
))

#let achievement(name) = [#text(
  name, 
  // style: "oblique",
  weight: "bold",
  size:8pt,
)]

= Chapter 1
Turing test: Human asks a machine and a human questions. Based only from the responses, determine which is the AI and which is the human. If the human is unable to distinguish or chooses wrongly, the machine passes the turing test

= Chapter 2
- Supervised learning is using labelled data to predict 
    - Subclass: Reinforcement learning is using unlabelled data and trial of error, then improve its performance from the results of each trial
- Unsupervised learning is unlabelled data in order to find patterns
- Sampling Bias: Non-representitive training data
- No Free Lunch Theorem: There is no one size fits all model

=== Bias Variance Tradeoff

Bias is when it is systematically incorrect for certain data points.
Variance is fluctuation in output when different inputs, essentially the flexibility.
- Balance of Precision and Recall, F1 Score = 2 \* (Precision \* Recall) / (Precision + Recall)
- AUC-ROC = Area under curve of TP rate against FP rate
#image("image.png", width: 50%)

=== To solve over/underfitting

- Choose a simpler model
    - Regularisation is a part of reducing model complexity by limiting some of the parameter coefficients close to 0
- Reduce parameters of model
- Gather more data
- Reduce outliers and errors in training data
- Solving underfitting is above inverse plus adding better features to training data

#figure(
    grid(
        columns: 2,
        gutter: 0mm,
        [#image("image2.png", width: 50%)],
        [#image("image3.png", width: 30%)],
    )
)
#figure(
    grid(
        columns: 2,
        gutter: 0mm,
        [#image("image4.png", width: 50%)],
        [#image("image5.png", width: 70%)],
    )
)
#image("image6.png", width: 50%)
#image("image7.png", width: 50%)
#image("image8.png", width: 50%)
#image("image9.png", width: 50%)
#image("image10.png", width: 50%)
