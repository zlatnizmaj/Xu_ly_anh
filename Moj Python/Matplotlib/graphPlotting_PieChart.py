import matplotlib.pyplot as plt
import re
import math

# The slices will be ordered and plotted counter-clokwise if startangle = 90
sizes = [175, 50, 25, 50]
total = sum(sizes)
print('TOTAL: ', total, '')

percentages = list(map(lambda  x: str((x/total * 1.00) * 100) + '%', sizes))
print('PERCENTAGES:',percentages)

backToFloat = list(map(lambda x: float(re.sub("%$", "", x)), percentages))
print('')

print('PERCENTAGES BACK TO FLOAT: ')
print(["{0:.2f}".format(btF) for btF in backToFloat])
print('')

print('SUM OF PERCENTAGES')
print(str(sum(backToFloat)))
print('')

labels = percentages
color = ['blue', 'red', 'green', 'orange']
patches, texts = plt.pie(sizes, counterclock= True, colors= color, explode= (0,0,0,0.1), startangle= -270)

plt.legend(patches, labels, loc= "best")
print(texts)


# Set aspect ratio to be equal so that pie is drawn as a circle.
plt.axis('equal')
plt.tight_layout()
plt.show()
