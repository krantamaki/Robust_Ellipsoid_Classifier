"""
Config file containing important constants like paths etc.
"""

subject_path = ""  # Path to the folder with all subjects
word2vec_path = ""

n_sensors = 204
n_timepoints = 40
n_subjects = 20
n_words = 60
n_reps = 18
n_averaged = 10

excluded_subjects = ["sub-10"]  # Measurements for subject 10 failed

stimuli = "picture"

distinct_labels = ["koira", "hevonen", "ankka", "kotka", "kissa", "leijona", "hiiri", "karhu", "lammas",
                   "selka", "kasi", "silma", "jalka", "korva", "suu", "varvas", "sormi", "nena",
                   "kirkko", "tie", "tehdas", "linna", "silta", "vankila", "torni", "kirjasto", "museo",
                   "kuningas", "sotilas", "poliisi", "vanki", "pappi", "opettaja", "laakari", "tuomari", "lapsi",
                   "joki", "saari", "meri", "puisto", "vuori", "kallio", "aalto", "pilvi", "pesa",
                   "kirja", "pallo", "saha", "sormus", "sakset", "lusikka", "haarukka", "lapio", "kampa",
                   "auto", "laiva", "juna", "vene", "bussi", "rekka"]

label_categories = {'animals': ["koira", "hevonen", "ankka", "kotka", "kissa", "leijona", "hiiri", "karhu", "lammas"],
                    'body': ["selka", "kasi", "silma", "jalka", "korva", "suu", "varvas", "sormi", "nena"],
                    'buildings': ["kirkko", "tie", "tehdas", "linna", "silta", "vankila", "torni", "kirjasto", "museo"],
                    'characters': ["kuningas", "sotilas", "poliisi", "vanki", "pappi", "opettaja", "laakari", "tuomari", "lapsi"],
                    'nature': ["joki", "saari", "meri", "puisto", "vuori", "kallio", "aalto", "pilvi", "pesa"],
                    'tools': ["kirja", "pallo", "saha", "sormus", "sakset", "lusikka", "haarukka", "lapio", "kampa"],
                    'vehicles': ["auto", "laiva", "juna", "vene", "bussi", "rekka"]}

word2vec_dim = 300
