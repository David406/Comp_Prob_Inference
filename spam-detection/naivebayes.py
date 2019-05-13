import sys
import os.path
import numpy as np
import collections

import util

USAGE = "%s <test data folder> <spam folder> <ham folder>"

def get_counts(file_list):
    """
    Computes counts for each word that occurs in the files in file_list.

    Inputs
    ------
    file_list : a list of filenames, suitable for use with open() or 
                util.get_words_in_file()

    Output
    ------
    A dict whose keys are words, and whose values are the number of files the
    key occurred in.
    """
    ### TODO: Write your code here
        
    # Count the number of files each word occurs in
    words_count_dict = collections.defaultdict(lambda: 0)
    for file in file_list:
        words_set = set(util.get_words_in_file(file))
        for word in words_set:
            words_count_dict[word] += 1
                
    return words_count_dict
    
    
    

def get_log_probabilities(file_list):
    """
    Computes log-frequencies for each word that occurs in the files in 
    file_list.

    Input
    -----
    file_list : a list of filenames, suitable for use with open() or 
                util.get_words_in_file()

    Output
    ------
    A dict whose keys are words, and whose values are the log of the smoothed
    estimate of the fraction of files the key occurred in.

    Hint
    ----
    The data structure util.DefaultDict will be useful to you here, as will the
    get_counts() helper above.
    """
    ### TODO: Write your code here
    words_count_dict = get_counts(file_list)
    
    num_files = len(file_list)
    
    # Notice that we set default value to be -log(num_files+2) since we are using smoothed prob.
    words_log_frequency = collections.defaultdict(lambda: -np.log(num_files+2))
    for word, count in words_count_dict.items():
        words_log_frequency[word] = np.log(count+1) - np.log(num_files+2)
        
    return words_log_frequency
    
    


def learn_distributions(file_lists_by_category):
    """
    Input
    -----
    A two-element list. The first element is a list of spam files, 
    and the second element is a list of ham (non-spam) files.

    Output
    ------
    (log_probabilities_by_category, log_prior_by_category)

    log_probabilities_by_category : A list whose first element is a smoothed
                                    estimate for log P(y=w_j|c=spam) (as a dict,
                                    just as in get_log_probabilities above), and
                                    whose second element is the same for c=ham.

    log_prior_by_category : A list of estimates for the log-probabilities for
                            each class:
                            [est. for log P(c=spam), est. for log P(c=ham)]
    """
    ### TODO: Write your code here
    spam_file_list = file_lists_by_category[0]
    ham_file_list = file_lists_by_category[1]
    
    log_probabilities_by_category = [None] * 2
    log_probabilities_by_category[0] = get_log_probabilities(spam_file_list)
    log_probabilities_by_category[1] = get_log_probabilities(ham_file_list)
    
    s1 = len(spam_file_list)
    s2 = len(ham_file_list)
    log_prior_by_category = [np.log(s1/(s1+s2)), np.log(s2/(s1+s2))]
    
    return (log_probabilities_by_category, log_prior_by_category)
    

def classify_email(email_filename,
                   log_probabilities_by_category,
                   log_prior_by_category):
    """
    Uses Naive Bayes classification to classify the email in the given file.

    Inputs
    ------
    email_filename : name of the file containing the email to be classified

    log_probabilities_by_category : See output of learn_distributions

    log_prior_by_category : See output of learn_distributions

    Output
    ------
    One of the labels in names.
    """
    ### TODO: Write your code here
    
    # Extract trained parameters
    log_s, log_h = log_prior_by_category
    log_q_dict = log_probabilities_by_category[0]
    log_p_dict = log_probabilities_by_category[1]
    
    # Read words from file
    email_words = util.get_words_in_file(email_filename)
    # Get all the words in the dictionary (all words in training samples)
    all_words = list(log_probabilities_by_category[0].keys())
    all_words.extend(list(log_probabilities_by_category[1].keys()))
    all_words = set(all_words)
    
    # Compute log(P(spam|Y) / P(ham|Y))
    odds_ratio = log_s - log_h
    for word in all_words:
        log_q_j = log_q_dict[word]
        log_p_j = log_p_dict[word]
        if word in email_words:
            odds_ratio += (log_q_j - log_p_j)
        else:
            odds_ratio += (np.log(1-np.exp(log_q_j)) - np.log(1-np.exp(log_p_j)))
            
    if odds_ratio >= 0:
        return "spam"
    else:
        return "ham"
    
    

def classify_emails(spam_files, ham_files, test_files):
    # DO NOT MODIFY -- used by the autograder
    log_probabilities_by_category, log_prior = \
        learn_distributions([spam_files, ham_files])
    estimated_labels = []
    for test_file in test_files:
        estimated_label = \
            classify_email(test_file, log_probabilities_by_category, log_prior)
        estimated_labels.append(estimated_label)
    return estimated_labels

def main():
    ### Read arguments
    if len(sys.argv) != 4:
        print(USAGE % sys.argv[0])
    testing_folder = sys.argv[1]
    (spam_folder, ham_folder) = sys.argv[2:4]

    ### Learn the distributions
    file_lists = []
    for folder in (spam_folder, ham_folder):
        file_lists.append(util.get_files_in_folder(folder))
    (log_probabilities_by_category, log_priors_by_category) = \
            learn_distributions(file_lists)

    # Here, columns and rows are indexed by 0 = 'spam' and 1 = 'ham'
    # rows correspond to true label, columns correspond to guessed label
    performance_measures = np.zeros([2,2])

    ### Classify and measure performance
    for filename in (util.get_files_in_folder(testing_folder)):
        ## Classify
        label = classify_email(filename,
                               log_probabilities_by_category,
                               log_priors_by_category)
        ## Measure performance
        # Use the filename to determine the true label
        base = os.path.basename(filename)
        true_index = ('ham' in base)
        guessed_index = (label == 'ham')
        performance_measures[true_index, guessed_index] += 1


        # Uncomment this line to see which files your classifier
        # gets right/wrong:
        #print("%s : %s" %(label, filename))

    template="You correctly classified %d out of %d spam emails, and %d out of %d ham emails."
    # Correct counts are on the diagonal
    correct = np.diag(performance_measures)
    # totals are obtained by summing across guessed labels
    totals = np.sum(performance_measures, 1)
    print(template % (correct[0],
                      totals[0],
                      correct[1],
                      totals[1]))

if __name__ == '__main__':
    main()
