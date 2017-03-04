* Numpy version is slower than TF version, therefore I didn't run it.
* TF version still only runs on CPU, which we can furthere improve on by using GPU
* I have run 7850 epoch in total, I reached local minimum around 4500 epoch. After that, loss keeps fluctuating without improment.
* Then best reward achieved is about 15, which is still not very good.
* At the end of the run, model store as epoch 7800 has the best performance, averaging mean reward 3.469991