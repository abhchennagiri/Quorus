import nltk
from helpers import clean_str
import linecache

#Define Global variables here
datafile = "./datasets/quora_duplicate_questions.tsv"
trainingFile = './datasets/training.full.tsv'
testFile = './datasets/test.full.tsv'
TEST_SIZE  = 0.1


def preprocess( datafile, MIN_LENGTH = 10, LIMIT = 59, header=True ):
    line_num = 150000
    lines = []
    max_len = 0
    longest_q = ""
    dups = 0
    sentences = 0
    skipped = 0
    skipped_dup = 0
    count = 0
    with open( datafile ) as f:
        for line in f:
            #count += 1
            #if count < 364000:
            #    continue
            if header == True:
                header = False
                continue
            #print line
            fields = line.strip('\n').split('\t')
            
            q1 = clean_str( fields[3] )
            q2 = clean_str( fields[4] )
            dup = fields[5]

            q1_len = len( q1.split() )
            q2_len = len( q2.split() )

            if q1_len > LIMIT or q2_len > LIMIT:
                skipped += 1
                if dup == '1':
                    skipped_dup += 1
                continue
    
            if q1_len + q2_len < MIN_LENGTH:
                skipped += 1
                if dup == '1':
                    skipped_dup += 1
                continue

            if q1_len > max_len:
                max_len = q1_len
                longest_q = q1


            if q2_len > max_len:  
                max_length = q2_len
                longest_q = q2

            if dup == '1':
                dups += 1
            
            if len( q1 ) == 0:
                q1 = "."
            
            if len( q2 ) == 0:
                q2 = "."

            lines.append( (q1, q2, dup ) )

            #print fields
            sentences += 1
            
                
    print "Longest question: %s (%d)" % (longest_q, max_len)
    print "duplicates: %d (%.2f)" % (dups, ((1.0 * dups) / sentences))
    print "skipped: %d (%d)" % (skipped, skipped_dup)
            
    return lines

                
def createTrainingData( trainingFile, lines ):
    index = -1 * int( len(lines) * TEST_SIZE )
    trainingLines = lines[:index]
    print "training: ", len(trainingLines)
    with open( trainingFile, 'w+' ) as f:
        for line in trainingLines:
            q1, q2, dup = line
            f.write('%s\t%s\t%s\n' %(dup, q1, q2))


def createTestData( testFile, lines ):
    index = -1 * int( len(lines) * TEST_SIZE )
    testLines = lines[index:]
    print "test: ", len(testLines)
    with open( testFile, 'w+' ) as f:
        for line in testLines:
            q1, q2, dup = line
            f.write( '%s\t%s\t%s\n' %(dup, q1, q2) )


def main():
    lines = preprocess( datafile )
    createTrainingData( trainingFile, lines )
    createTestData( testFile, lines )

if __name__ == '__main__':
    main()

