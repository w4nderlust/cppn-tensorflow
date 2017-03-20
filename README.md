# cppn-tensorflow

Simplified implementation of [Compositional Pattern Producing Network](https://en.wikipedia.org/wiki/Compositional_pattern-producing_network) in TensorFlow for the purpose of abstract art generation and for future deep learning work in generative algorithms.

![Rainbow](http://www.w4nderlu.st/content/2-projects/15-rppn/img0_2880_1800.png)

![Dragon Eye](http://www.w4nderlu.st/content/2-projects/15-rppn/img1_2880_1800.png)

See Otoro's blog post at [blog.otoro.net](http://blog.otoro.net/2016/03/25/generating-abstract-patterns-with-tensorflow/) for more details.
See my blogpost at [w4nderlu.st](http://www.w4nderlu.st/projects/rppn) for details on RPPN.

![Morphing](https://cdn.rawgit.com/hardmaru/cppn-tensorflow/master/examples/cppn.gif)
![Morphing](https://cdn.rawgit.com/hardmaru/cppn-tensorflow/master/examples/output.gif)

My contribution:
- porting to Python 3 and Tensorflow 1.0
- video generation
- added a new model, RPPN (Recursive Pattern Producing Network)

Requirements:
- TensorFlow 1.0.0+
- imageio for image generation
- ffmpeg for video generation

# License

BSD - images2gif.py

MIT - everything else
