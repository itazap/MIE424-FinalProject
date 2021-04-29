# MIE424-FinalProject

**Type of project:** A. Implementation

**Tentative Title:** Evaluating Fairness Using Constraint Relaxation to Solve Convex Functions

**Members:** Ita Zaporozhets, Farhan Wadia, Chris Palumbo, Merih Atasoy 

**Most relevant papers from the literature (or open-source projects from the web):**

M. Donini, L. Oneto, S. Ben-David. "Empirical Risk Minimization under Fairness Constraints", 2018. [Online]. Available: https://arxiv.org/pdf/1802.08626.pdf. [Accessed: 29- Mar- 2021].

Y. Wu, L. Zhang and X. Wu, "On Convexity and Bounds of Fairness-aware Classification", Csce.uark.edu, 2019. [Online]. Available: http://www.csce.uark.edu/~xintaowu/publ/www19.pdf. [Accessed: 29- Mar- 2021].

M. Lohaus and M. Perrot, "Too Relaxed to Be Fair", 2020. [Online]. Available: http://proceedings.mlr.press/v119/lohaus20a/lohaus20a.pdf. [Accessed: 29- Mar- 2021].

# Noteable code that has been reused

Our project has used/adapted code from Michael Lohous’ SearchFair repository, found at: https://github.com/mlohaus/SearchFair.

The citation for this SearchFair repository is:

@inproceedings{lohaus2020,
  title={Too Relaxed to Be Fair},
  author={Lohaus, Michael and Perrot, Micha{\"e}l and von Luxburg, Ulrike},
  booktitle={International Conference on Machine Learning},
  year={2020}
}

# Setup and Run

*We suggest using Google Colab for running the notebook files.

Notebooks to run the various result tests are in the main project directory, along with the Adult Dataset exploration notebook.

The source code for the different class and functions can be found in the 'src' folder, and must be uploaded to Google Colab using the same folder name and hierarchy. 

The dataset installation command simply clones Michael Lohous’ Github repository within each of the necessary notebooks. Running this command will clone the entire repository into the Colab workspace and will provide the up-to-date data. 

Next, the reqiured packages installed can be found in requirements.txt.

Finally, the result notebooks can be uploaded and opened in Google Colab to run by either individually running each block, or by running the entire file at once. 

 
# Project Description

The problem of focus is classification under fairness constraints, based on the literature from Michael Lohaus, Michael Perrot, and Ulrike von Luxburg titled ‘Too Relaxed To Be Fair.’ The paper builds upon previous work to address the challenge of training a classifier that is not biased against a group of individuals using constrained optimization. The initial approach to the problem by Donini (2018) is initially formulated as a nonconvex, nonsmooth minimization problem. Due to the difficulty in optimizing these types of problems, Donini used a simple linear relaxation of constraints to transform into a convex optimization problem. Wu (2019) gained improvements in results by attempting to address a transformation to convexity using lower-upper relaxation and surrogate functions. Finally, Lohaus (2020) introduced a novel approach by reformulating a convex function through a proposed SearchFair algorithm yielding the best overall results. 

Our team plans to recreate the past four years of advancements, by going through and implementing each iteration of the problem. First, our team plans on recreating the initial proposed convex formulation by Donini using a linear relaxation of the problem. Next, our team plans to recreate Wu’s lower-upper relaxation and surrogate functions. Finally, the team plans to implement Lohaus’ SearchFair algorithm for relaxation. Each iteration will be tested and compared to results of a baseline linear classifier that disregard fairness. The function being optimized is the empirical difference of demographic parity (DDP) with a convex regularization term, and a convex approximation of a signed fairness constraint. SearchFair was tested on 6 real-world datasets and compared to 5 baselines. Our team will attempt to recreate these results for the Adult dataset, which makes a binary prediction of income levels based on 14 attributes such as education, race, sex, marital status, etc. This dataset was used by all three papers and would apply easily in extending the problem to test SearchFair (time permitting) with multiple sensitive attributes, rather than a single sensitive attribute. 



