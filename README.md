# Unsupervised Image normalization: Atlas
![](https://img.shields.io/badge/<Implementation>-<population_representing>-<success>)
![](https://img.shields.io/badge/<Implementation>-<normalize_input>-<success>)

[![ko-fi](https://www.ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/R5R11K2H4)


## Getting Started

Simply open the jupyter notebook and see how some demo on pictures that we uploaded with this repository

### Prerequisites

What things you need to install the software and how to install them

```
scikit-image
matplotlib
numpy
pytorch
notebook
scipy
```

### Installing

Here are the steps to follow

```
conda install environment.yml
conda activate atlas
```

## Running the tests

Go over main.py, and COMMENT OUT the current script and UNCOMMENT OUT the DEMO part then:

```
python3 main.py
```

### Break down into end to end tests

Our tests are working on three different sets of images. Here are some examples of results and their explanation.

#### Implementation of demon 0
- demons are on are scattererd on the contour of $S$.
- deformation are rigid, so one direction for all pixels
- iterative $\phi_n$ given by the affine transform, so explicit
- magnitude of force same, but correctness still assured since the number of support to exert force supposedly diminish when shapes overlap

DISCLAIMER: the maximum generality we allow ourselves is to deal only with disks of different direction, and allow ourselves only rigid transform plus white background

![](https://github.com/miki998/image_registration-maxwell_demons/blob/master/readme_images/demon0.png)

## Deployment

None yet, you can do some pull requests to me

## Built With

* [python3](https://www.python.org/download/releases/3.0/) - The web framework used

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Authors
Michael Chan
## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details

## Acknowledgments
...








