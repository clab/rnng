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

def get_dict(lines):
    output = [] 
    for line in lines:
        #print 'curr line', line_strip
        line_strip = line.rstrip()
        #print 'length of the sentence', len(line_strip)
        for i in range(len(line_strip)):
            if i == 0:
                assert line_strip[i] == '('
            if line_strip[i] == '(' and not(is_next_open_bracket(line_strip, i)): # fulfilling this condition means this is a terminal symbol
                output.append(get_between_brackets(line_strip, i))
        #print 'output:',output
    words_dict = {} 
    for terminal in output:
        terminal_split = terminal.split()
        assert len(terminal_split) == 2 # each terminal contains a POS tag and word        
        if not(terminal_split[1] in words_dict):
            words_dict[terminal_split[1]] = 1
        else:
            words_dict[terminal_split[1]] = words_dict[terminal_split[1]] + 1
    words_list = []
    for item in words_dict:
        if words_dict[item] > 1:
            words_list.append(item) 
    return words_list 

if __name__ == '__main__':
    input_file = open('train.02-21', 'r')
    lines = input_file.readlines()
    words_list = get_dict(lines) 
    #print 'number of words', len(words_list)
    for word in words_list:
        print word
    input_file.close() 
