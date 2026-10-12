# Third-party data and dependency notices

## Research method

The model follows the collaborative-filtering method described in Xiao Zhou, “Collaborative Filtering on Keywords Recommendation for Clinical Trial Records,” Stanford CS229, 2020: https://cs229.stanford.edu/proj2020spr/report/Zhou.pdf.

The model and evaluation modules in `ml/` are implemented within this project. The paper is cited as the source of the research method; it is not a claim that the algorithm itself is original to this team.

## Dataset

The dataset downloader uses https://huggingface.co/datasets/louisbrulenaudet/clinical-trials, based on ClinicalTrials.gov. The dataset's metadata and README have historically presented inconsistent license information. Review the current source repository's terms before redistributing any downloaded records. This project does not commit the downloaded source dataset.

## Software dependencies

Third-party Python packages required to run the project are listed in `requirements.txt`; they are installed from their respective package distributions and are not copied into this repository.
