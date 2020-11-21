# sequence1 = "((((((((((((((.[[[[[[..))))).....]]]]]]........)))))...))))"
# sequence2 = "((([[(..]]))))"
# sequence3 = "((([[(..]]((..[[[...))]]]))))"



# parentheses = []
# brackets = []

# index = 1

# with open("contact_list.txt", 'w') as fwrite:
#     for i in sequence3:
#         if i == '(':
#             parentheses.append(index)
#         elif i == '[':
#             brackets.append(index)
#         if i == ')':
#             #print("%s %s" %(parentheses.pop(), index))
#             fwrite.write("%s %s\n" %(parentheses.pop(), index))
#         if i == ']':
#             #print("%s %s" %(brackets.pop(), index))
#             fwrite.write("%s %s\n" %(brackets.pop(), index))
#         index += 1

import numpy as np
a = np.loadtxt('contact_list.txt')
print(len(a[0]))