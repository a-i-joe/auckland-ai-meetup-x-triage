# ![X-Triage logo](https://a-i-joe.github.io/auckland-ai-meetup-x-triage/X-Triage-banner.png "X-Triage (auckland-ai-meetup-x-triage)")

X-Triage is a collaboration by the [Auckland Artificial Intelligence Programming Meetup](https://www.meetup.com/Auckland-AI-Meetup/). We aim to advance research in the area of rapid x-ray triage.

## Semantics

A "negative" label indicates an x-ray which has no abnormality (it's healthy).

A "positive" label indicates the presence of an abnormality.

## Goals

**1.** For our first initiative, our hypothesis is that a significant number of normal chest x-rays can be filtered out of the triage process by a machine learning model, even if we avoid false negatives with a certainty approaching 100%. 

This could potentially have huge implications in emergency situations around the world - often there is no expert reviewer at hand and if even a small proportion of x-rays could be cleared immediately then that would be helpful.

**1a.** A secondary objective is to see how quickly our results improve as sample count increases.

### Success Measurements

**1.**

- Percentage of normal x-rays identified when false-negative tolerance approaches `0`. That's a mouthful, so let's call it the **'safe set**'.  
  *What proportion of the healthy people can we filter from triage, without accidentally clearing people who need to be seen?*
- Area Under the Curve  
  *How good, overall, is the model for varied false negative/positive tolerance settings?*

**1a.**

- Chart the **safe set** percentage (y) against the number of training examples (x).
- Chart the AUC (y) against the number of training examples (x).

## Source data

Dr. Greg Tarr has commended the following data sources to us.

In total, there are around 8,000 chest x-ray films, which have been completely de-identified and have both the de-identified clinical reports and have been labelled as "normal" or having other findings.

The image files are in DICOM format, which is JPEG + a medical header with scan information. These images can be viewed in a DICOM viewer like [RadiAnt (Win)](www.radiantviewer.com) or [Osirix (Mac)](www.osirix-viewer.com).

FAQ: https://openi.nlm.nih.gov/faq.php

### Larger dataset:  
https://openi.nlm.nih.gov/imgs/collections/NLMCXR_dcm.tgz

For this dataset, the labels are unfortunately not included in the download. They're available through a restful endpoint, e.g.  https://openi.nlm.nih.gov/retrieve.php?query=&coll=iu
They limit the queries to 30 results at a time, so it would be helpful to create a tool which can iterate to retreive all the labels for a desired set of images. There's a readme on the restful API, here: https://openi.nlm.nih.gov/services.php?it=xg#params . @rasutt has contributed a tool which you can use to download these ([XRayData/](https://github.com/a-i-joe/auckland-ai-meetup-x-triage/tree/master/XRayData)).

### Smaller dataset 1:

https://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip

### Smaller dataset 2:

https://openi.nlm.nih.gov/imgs/collections/ChinaSet_AllFiles.zip

## Delivering our findings

Dr. Greg Tarr is medical doctor, which provides us an excellent connection to the communities who might find our research useful.

Dr. Tarr hopes to present our findings to the Radiological Society of North America in Chicago in November. This is the biggest radiology conference in the world with around 50,000 participants. It's pretty competitive but if we have good results, Greg thinks we'd have a good chance at getting in.

## How to contribute

Please submit your work to this repository using the standard [github flow](https://guides.github.com/introduction/flow/) (fork and pull-request).

### Guidelines

- Add a folder for your experiment.
- Submit the full code (or steps) required to repeat your experiment.
- Wherever possible, provide infrastructure requirements. For example; machine specifications, software required, python version, etc. Ideally, provide infrastructure as code for spinning up an identical environment in a mainstream cloud environment (e.g. terraform/cloudformation defining an AWS setup with packer/vagrant/a script/etc for installing the right software). The overriding principal is to describe everything needed to reproduce your results.
- Please review and consider accepting pull requests from others to improve your experiments. This is a collaboration.
- Include a README.md file in your folder, explaining the experiment and your results so far, and keep it updated as the experiment improves.
- Also describe your process for splitting training, validation and testing data. Final solutions will need to show due-diligence, such as reserved testing data and 10 × 10-fold validation, however don't let that scare you off having a try. We can refine the experiments together.
- Please don't commit the actual data to the repository. Rather, explain which datasets to use and how to arrange them for your solution. This is the easiest way to ensure we meet the terms and conditions for all the datasets.

### License

The repository uses the MIT license, which is permissive, but requires attribution and prevents us being liable for it's use.

### Communication

- Use the github issues in this repository, or the [meetup board](https://www.meetup.com/Auckland-AI-Meetup/messages/boards/), for discussing experiments and ideas.
- You can [message Dr Greg Tarr through meetup.com](https://www.meetup.com/Auckland-AI-Meetup/members/115831142/) to consult on the project from a medical perspective.
