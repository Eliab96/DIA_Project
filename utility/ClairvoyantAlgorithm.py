import math
import pandas as pd

pd.set_option('precision', 3)

inf = -math.inf

'''
    Algoritmo knapsack, semplicemente passandogli una tabella di valori trova quali budged allocare per ottenere il
    massimo ricavo. 
'''


def ClairvoyantKnapsack(table):
    rows = len(table)
    cols = len(table[0])

    # Elimino i valori negativi ponendoli a meno infinito
    for row in range(rows):
        for col in range(cols):
            if table[row][col] < 0:
                table[row][col] = inf

    optTable = [[] for row in range(rows)]
    # Mi basta sapere il budget di due sottocampagne, la terza semplicemente faccio maxbudget - i due budget
    optIndexes = [[] for row in range(rows - 1)]

    # Copio la prima colonna
    for col in range(cols):
        optTable[0].append(table[0][col])

    # Provo tutte le possibili combinazioni di reward con l'incremento di budget
    for row in range(1, rows):
        for col in range(cols):
            temp = []
            for col2 in range(col + 1):
                tempVal = table[row][col2] + optTable[row - 1][col - col2]
                temp.append(tempVal)
            maxValue = max(temp)
            optTable[row].append(maxValue)
            optIndexes[row - 1].append(temp.index(maxValue))

    # Valore dell'optimal reward
    optValue = max(optTable[rows - 1])
    optCol = optTable[rows - 1].index(optValue)
    assignments = [0 for r in range(rows)]
    for row in range(rows - 1, 0, -1):
        subCol = optIndexes[row - 1][optCol]
        assignments[row] = subCol
        optCol -= subCol
    assignments[0] = optCol
    # Restituisco i 3 valori di budget per ogni sottocampagna
    return assignments


'''
    Crea un dataframe con i valori dlle clickfunction
'''


def GetDF(table, assignments, columns):
    def highlight(s, map):
        a = next(map)
        attr = 'background-color: yellow'
        return [attr if v == a else '' for v in range(len(s))]

    df = pd.DataFrame(data=table,
                      index=pd.Index(["C" + str(i) for i in range(len(table))]),
                      columns=columns
                      )
    df.insert(len(table[0]), 'Budget', assignments)
    df = df.style.apply(highlight, map=iter(assignments), axis=1)
    return df
