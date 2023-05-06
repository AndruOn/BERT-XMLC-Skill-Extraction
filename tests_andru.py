from pybert.preprocessing.preprocessor import EnglishPreProcessor
from pybert.configs.basic_config import config

from pybert.io.task_data import TaskData


import pickle
import pprint
import pandas as pd

#What is the format the training dataset stored in piclke:
#obj = pickle.load(open("pybert/dataset/processed/job_dataset.test.pkl", "rb"))
#with open("job_dataset_tes_pkl_decompressed.txt", "a") as f:
#        pprint.pprint(obj, stream=f)

#TODO: 
# - save as csv
# - save as check taht it runs with the read_data func
# - gotta make a "preprocessed dataset" because that 
#       will be the input of the model so domain similarity on this
# can i use read_data or just recreate the a functdoing it
# problem: the preprocessor works per sentence and outputs broken sentences
def read_data(self,raw_data_path,preprocessor = None,is_train=True):
        '''
        :param raw_data_path:
        :param skip_header:
        :param preprocessor:
        :return:
        '''
        targets, sentences = [], []
        data = pd.read_csv(raw_data_path)
        for row in data.values:
                if is_train:
                        target = row[2:]
                else:
                        target = [-1,-1,-1,-1,-1,-1]
                sentence = str(row[1])
                if preprocessor:
                        sentence = preprocessor(sentence)
                if sentence:
                        targets.append(target)
                        sentences.append(sentence)
        return targets,sentences

#PREPROCESSING IN PAPER: lower-casing, stopword removal, and rare word remova

role = "?:UPPER:?X You have several years of experience in digital (customer) data analysis. You have knowledge/experience of digital marketing, e-commerce and e-care, including web analytics, social media and conversion marketing. You have experience with the use of analytical tools (preferably Adobe Analytics). Knowledge of tag management, Adobe Experience Manager and Adobe Audience Manager is a strong asset. You are fluent in PowerPoint and Excel and have a basic knowledge of web technologies. You have good analytical skills. You are a critical thinker and a good problem solver. You are a real team player who can work well with colleagues from different teams, both from a business and a more technical background. You have strong communication and presentation skills. You love to make data transparent for business and enjoy taking on less experienced colleagues. You are willing to further develop skills in a team-oriented, fast-changing environment."

company_info_NTU = "World's Top Young University\nA research-intensive public university, Nanyang Technological University, Singapore (NTU Singapore) has 33,000 undergraduate and postgraduate students in the colleges of Engineering, Business, Science, and Humanities, Arts and Social Sciences, and its Interdisciplinary Graduate School. NTU\u2019s Lee Kong Chian School of Medicine was established jointly with Imperial College London.\nMeteoric rise in international academic reputation\u00a0?\nIn 2018, NTU was placed 12th globally in the Quacquarelli Symonds (QS) World University Rankings. It was also ranked the world\u2019s best young university (under 50 years old) by QS for the fifth consecutive year. In addition, NTU was named the world\u2019s fastest rising young university by Times Higher Education in 2015.\nIn engineering and technology, NTU is ranked 5th worldwide in the QS World University Rankings by Subject 2018. With six schools, NTU\u2019s College of Engineering is among the top nine globally for research output and the 5th most cited in the world (Essential Science Indicators 2017).\u00a0?\nMirroring this success is the College of Science, whose young chemistry department is ranked 10th among universities in the Nature Index 2018. Boosted by research at the Lee Kong Chian School of Medicine, NTU is also strengthening its foothold in areas such as biomedicine and life sciences.\nThe well-established Nanyang Business School is regularly featured among the leading business schools in Asia, with its MBA programme consistently rated top in Singapore since 2004 by The Economist.\nInnovative learning\nIn higher education, NTU is driving new pedagogies so that millennials can learn more effectively in this digital age. Part of NTU's education strategy is the flipped classroom model of learning. The centrepiece of this new way of learning is The Hive, a groundbreaking learning facility that has been described by CNN as having redefined the traditional classroom.\u00a0\nInnovative education initiatives also include signature programmes such as Renaissance Engineering Programme, CN Yang Scholars Programme, NTU-University Scholars Programme. Designed for high-achieving students, these programmes offer a multidisciplinary curriculum, guidance by top faculty, interdisciplinary and intensive research opportunities, overseas exposure, as well as dialogues with world-class scientists and industry leaders.?\nSet up jointly with Imperial College London, NTU's medical school, is grooming a new generation of doctors to meet the healthcare needs of Singapore.\nResearch impact\nKnown for research excellence and technological innovation, NTU leads the top Asian universities in normalised research citation impact (Clarivate Analytics\u2019 InCites 2016). In the 2018\u00a0Nature Index, NTU is placed 29th among the world's universities and first in Singapore.\nNTU's five-year strategic plan, NTU 2020, builds on strong foundations of NTU 2015 and aims to propel NTU to greater heights of research excellence. The plan focuses on five key research thrusts \u2013 Sustainable Earth, Global Asia, Secure Community, Healthy Society and Future Learning. The areas leverage NTU\u2019s diverse strengths, particularly its longstanding expertise in engineering, business and education, and the interfaces these have with various disciplines such as in healthcare, science and the humanities. NTU\u2019s sustainability initiatives have clinched significant competitive research funding and the university is already a global leader in this area.\n"


s = "DataAnalysis/jobads_dataset_test.csv"

#preprocessor = EnglishPreProcessor() #min_len=2,stopwords_path=None
#cleaned = preprocessor.__call__(s)

data_preprocessor = TaskData()
cleaned = data_preprocessor.read_data("DataAnalysis/resumes_dataset_test.csv",
                                            preprocessor=EnglishPreProcessor(),
                                            is_train=True)
with open("DataAnalysis/resumes_preprocessed_test.txt", "w") as f:
        pprint.pprint(cleaned, stream=f)

print("raw:\n",s)
print("preprocessed:\n",cleaned)



