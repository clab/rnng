import sys
import get_dictionary

# tokens is a list of tokens, so no need to split it again
def unkify(tokens, words_dict):
    final = []
    for token in tokens:
        # only process the train singletons and unknown words
        if len(token.rstrip()) == 0:
            final.append('UNK')
        elif not(token.rstrip() in words_dict):
            numCaps = 0
            hasDigit = False
            hasDash = False
            hasLower = False
            for char in token.rstrip():
                if char.isdigit():
                    hasDigit = True
                elif char == '-':
                    hasDash = True
                elif char.isalpha():
                    if char.islower():
                        hasLower = True
                    elif char.isupper():
                        numCaps += 1
            result = 'UNK'
            lower = token.rstrip().lower()
            ch0 = token.rstrip()[0]
            if ch0.isupper():
                if numCaps == 1:
                    result = result + '-INITC'    
                    if lower in words_dict:
                        result = result + '-KNOWNLC'
                else:
                    result = result + '-CAPS'
            elif not(ch0.isalpha()) and numCaps > 0:
                result = result + '-CAPS'
            elif hasLower:
                result = result + '-LC'
            if hasDigit:
                result = result + '-NUM'
            if hasDash:
                result = result + '-DASH' 
            if lower[-1] == 's' and len(lower) >= 3:
                ch2 = lower[-2]
                if not(ch2 == 's') and not(ch2 == 'i') and not(ch2 == 'u'):
                    result = result + '-s'
            elif len(lower) >= 5 and not(hasDash) and not(hasDigit and numCaps > 0):
                if lower[-2:] == 'ed':
                    result = result + '-ed'
                elif lower[-3:] == 'ing':
                    result = result + '-ing'
                elif lower[-3:] == 'ion':
                    result = result + '-ion'
                elif lower[-2:] == 'er':
                    result = result + '-er'            
                elif lower[-3:] == 'est':
                    result = result + '-est'
                elif lower[-2:] == 'ly':
                    result = result + '-ly'
                elif lower[-3:] == 'ity':
                    result = result + '-ity'
                elif lower[-1] == 'y':
                    result = result + '-y'
                elif lower[-2:] == 'al':
                    result = result + '-al'
            final.append(result)
        else:
            final.append(token.rstrip())
    return final 

def is_next_open_bracket(line, start_idx):
    for char in line[(start_idx + 1):]:
        if char == '(':
            return True
        elif char == ')':
            return False
    raise IndexError('Bracket possibly not balanced, open bracket not followed by closed bracket')    

def get_between_brackets(line, start_idx):
    output = []
    for char in line[(start_idx + 1):]:
        if char == ')':
            break
        assert not(char == '(')
        output.append(char)    
    return ''.join(output)

# start_idx = open bracket
#def skip_terminals(line, start_idx):
#    line_end_idx = len(line) - 1
#    for i in range(start_idx + 1, line_end_idx):
#        if line[i] == ')':
#            assert line[i + 1] == ' ' 
#            return (i + 2)
#    raise IndexError('No close bracket found in a terminal')

def get_tags_tokens_lowercase(line):
    output = []
    #print 'curr line', line_strip
    line_strip = line.rstrip()
    #print 'length of the sentence', len(line_strip)
    for i in range(len(line_strip)):
        if i == 0:
            assert line_strip[i] == '('    
        if line_strip[i] == '(' and not(is_next_open_bracket(line_strip, i)): # fulfilling this condition means this is a terminal symbol
            output.append(get_between_brackets(line_strip, i))
    #print 'output:',output
    output_tags = []
    output_tokens = []
    output_lowercase = []
    for terminal in output:
        terminal_split = terminal.split()
        assert len(terminal_split) == 2 # each terminal contains a POS tag and word        
        output_tags.append(terminal_split[0])
        output_tokens.append(terminal_split[1])
        output_lowercase.append(terminal_split[1].lower())
    return [output_tags, output_tokens, output_lowercase]    

def get_nonterminal(line, start_idx):
    assert line[start_idx] == '(' # make sure it's an open bracket
    output = []
    for char in line[(start_idx + 1):]:
        if char == ' ':
            break
        assert not(char == '(') and not(char == ')')
        output.append(char)
    return ''.join(output)


def get_actions(line):
    output_actions = []
    line_strip = line.rstrip()
    i = 0
    max_idx = (len(line_strip) - 1)
    while i <= max_idx:
        assert line_strip[i] == '(' or line_strip[i] == ')'
        if line_strip[i] == '(':
            if is_next_open_bracket(line_strip, i): # open non-terminal
                curr_NT = get_nonterminal(line_strip, i)
                output_actions.append('NT(' + curr_NT + ')')
                i += 1  
                while line_strip[i] != '(': # get the next open bracket, which may be a terminal or another non-terminal
                    i += 1
            else: # it's a terminal symbol
                output_actions.append('SHIFT')
                while line_strip[i] != ')':
                    i += 1
                i += 1
                while line_strip[i] != ')' and line_strip[i] != '(':
                    i += 1
        else:
             output_actions.append('REDUCE')
             if i == max_idx:
                 break
             i += 1
             while line_strip[i] != ')' and line_strip[i] != '(':
                 i += 1
    assert i == max_idx  
    return output_actions

def main():
    if len(sys.argv) != 3:
        raise NotImplementedError('Program only takes two arguments:  train file and dev file (for vocabulary mapping purposes)')
    train_file = open(sys.argv[1], 'r')
    lines = train_file.readlines()
    train_file.close()
    dev_file = open(sys.argv[2], 'r')
    dev_lines = dev_file.readlines()
    dev_file.close()
    words_list = get_dictionary.get_dict(lines) 
    line_ctr = 0
    # get the oracle for the train file
    for line in dev_lines:
        line_ctr += 1
        # assert that the parenthesis are balanced
        if line.count('(') != line.count(')'):
            raise NotImplementedError('Unbalanced number of parenthesis in line ' + str(line_ctr)) 
        # first line: the bracketed tree itself itself 
        print '# ' + line.rstrip()
        tags, tokens, lowercase = get_tags_tokens_lowercase(line)
        assert len(tags) == len(tokens)
        assert len(tokens) == len(lowercase)
        #print ' '.join(tags)
        print ' '.join(tokens)
        #print ' '.join(lowercase)
        unkified = unkify(tokens, words_list)    
        print ' '.join(unkified)
        output_actions = get_actions(line)
        for action in output_actions:
            print action
        print ''
    

if __name__ == "__main__":
    main()
