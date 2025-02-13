# CloneMe.jl

I downloaded "all" my livestream transcripts and I am training some basic language models on them. I am using Julia and `Flux` for gradients and optimization.

The model currently is a basic character level MLP that predicts the next letter from a sequence of previous. 

I also have another repo where I worked through more of the math following the "Neural Network Zero to Hero" series:
https://github.com/anandijain/nn_zero_to_hero


todo: 
* make the batching work so that we do the padding basically ;;;a -> b etc 
* write a grid searchy thing to see if larger models on gpu end up being actually faster 
* evaluate the test loss using a bigram model
* attention

done:
* log some generated text during training to see if it seems to meaningfully improve


current best score test loss: 1.3629686f0