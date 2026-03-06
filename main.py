import numpy as np
import matplotlib.pyplot as plt


def initPopBinary(npop, clen):
    pop = np.random.randint(0, 2, (npop, clen))
    return pop


def oneMaxI(chromosome):
    return sum(chromosome)

def oneMaxP(pop, npop):
    popFit = np.zeros(npop)
    for i in range(npop):
        popFit[i] = oneMaxI(pop[i])
    return popFit

def selectionProb(popFit):
    totalFitness = sum(popFit)
    if totalFitness == 0:
        return [1/len(popFit)] * len(popFit)
    probs = [i/totalFitness for i in popFit]
    return probs


def cumProb(probs):
    cprob = np.zeros_like(probs)
    cprob[0] = probs[0]
    for i in range(1, len(probs)):
        cprob[i] = cprob[i-1] + probs[i]
    if len(cprob) > 0:
        cprob[-1] = 1.0
    return cprob


def rouletteWheel(cprob):
    r = np.random.random()
    for i in range(len(cprob)):
        if r <= cprob[i]:
            return i
    return len(cprob) - 1

def rouletteSelect(cprob, pop):
    twoParents = np.zeros((2, np.size(pop, 1)))
    for i in range(2):
        indx = rouletteWheel(cprob)
        twoParents[i, :] = pop[indx, :]
    return twoParents

def binCross(twoParents, pcross, clen):
    twoChildren = twoParents.copy()
    
    if np.random.random() < pcross and clen > 1:
        point = np.random.randint(1, clen)
        twoChildren[0, :point] = twoParents[0, :point]
        twoChildren[0, point:] = twoParents[1, point:]
        twoChildren[1, :point] = twoParents[1, :point]
        twoChildren[1, point:] = twoParents[0, point:]
    
    return twoChildren


def binMutate(individual, pmute, clen):
    mutatedInd = individual.copy()
    
    for i in range(clen):
        if np.random.random() < pmute:
            mutatedInd[i] = 1 - mutatedInd[i]
    
    return mutatedInd


def runBinGA(npop, clen, ngen, pcross, pmute):
    
    pop = initPopBinary(npop, clen)
    

    bestHist = []
    avgHist = []
    

    for gen in range(ngen):
     
        fitness = oneMaxP(pop, npop)
        

        bestHist.append(np.max(fitness))
        avgHist.append(np.mean(fitness))
        

        probs = selectionProb(fitness)
        cprob = cumProb(probs)
        

        eliteIndices = np.argsort(fitness)[-2:][::-1]
        elites = pop[eliteIndices].copy()
        
        newPop = elites.copy()
 
        while len(newPop) < npop:
            parents = rouletteSelect(cprob, pop)
            children = binCross(parents, pcross, clen)
            
            for i in range(2):
                children[i] = binMutate(children[i], pmute, clen)
            
            newPop = np.vstack([newPop, children])
        
        pop = newPop[:npop]
    
    return pop, bestHist, avgHist

def runMultipleRuns(num_runs=5):


    print("Running  5 times with diffren random seeds ")
    
    
    allBestHistories = []
    allAvgHistories = []
    
    for run in range(num_runs):
        seed = run * 10
        np.random.seed(seed)
        
        print(f"\nRun {run+1} (Seed: {seed}) ")
        
        pop, bestHist, avgHist = runBinGA(
            npop=20, clen=5, ngen=100, pcross=0.6, pmute=0.05
        )
        
        allBestHistories.append(bestHist)
        allAvgHistories.append(avgHist)
        
        print(f"Final Best Fitness: {bestHist[-1]}/5")
        print(f"Final Avg Fitness: {avgHist[-1]:.2f}")
    

   
    print("\nSUMMARY OF 5 RUNS")
   
    print("Run | Best Fitness | Avg Fitness")
   
    for i in range(num_runs):
        print(f"{i+1:3d} |      {allBestHistories[i][-1]}      |    {allAvgHistories[i][-1]:.2f}")
    
    return allBestHistories, allAvgHistories


def printFinalPopulation():

    print("FINAL POPULATION (LAST RUN)")
   
    
    seed = 40  
    np.random.seed(seed)
    
    final_pop, _, _ = runBinGA(
        npop=20, clen=5, ngen=100, pcross=0.6, pmute=0.05
    )
    
    print(f"\nRun with Seed: {seed}")

    print(f"{'Index':<6} {'Chromosome':<10} {'Fitness':<10}")
    print("-" * 40)
    
    for i in range(len(final_pop)):
     
        chromosome = ''.join([str(int(final_pop[i][j])) for j in range(len(final_pop[i]))])
        fitness_val = oneMaxI(final_pop[i])
        print(f"{i+1:<6} {chromosome:<10} {fitness_val:<10}")
    
    print("-" * 40)
    
 
    all_fitness = [oneMaxI(ind) for ind in final_pop]
    print(f"Best Fitness: {max(all_fitness)}/5")
    print(f"Average Fitness: {np.mean(all_fitness):.2f}")
    print(f"Number of optimal solutions (11111): {sum(1 for f in all_fitness if f == 5)}/{len(final_pop)}")

# ========== PLOT WITH AND WITHOUT ELITISM ==========
def plotElitismComparison():

    print("\n" + "=" * 60)
    print("GENERATING PLOT FOR REPORT (WITH vs WITHOUT ELITISM)")
    print("=" * 60)
    
 
    def runBinGA_without_elitism(npop, clen, ngen, pcross, pmute):
        pop = initPopBinary(npop, clen)
        bestHist = []
        avgHist = []
        
        for gen in range(ngen):
            fitness = oneMaxP(pop, npop)
            bestHist.append(np.max(fitness))
            avgHist.append(np.mean(fitness))
            
            probs = selectionProb(fitness)
            cprob = cumProb(probs)
            
            
            newPop = []
            
            while len(newPop) < npop:
                parents = rouletteSelect(cprob, pop)
                children = binCross(parents, pcross, clen)
                
                for i in range(2):
                    children[i] = binMutate(children[i], pmute, clen)
                
                if len(newPop) == 0:
                    newPop = children.copy()
                else:
                    newPop = np.vstack([newPop, children])
            
            pop = newPop[:npop]
        
        return pop, bestHist, avgHist
    
    #  WITH Elitism
    np.random.seed(42)
    print("\n--- Running WITH Elitism ---")
    _, best_with, avg_with = runBinGA(
        npop=20, clen=5, ngen=100, pcross=0.6, pmute=0.05
    )
    
    
    np.random.seed(42)
    print("--- Running WITHOUT Elitism ---")
    _, best_without, avg_without = runBinGA_without_elitism(
        npop=20, clen=5, ngen=100, pcross=0.6, pmute=0.05
    )
    

    plt.figure(figsize=(14, 6))
    generations = range(100)

    plt.subplot(1, 2, 1)
    plt.plot(generations, best_with, 'b-', linewidth=2, label='With Elitism')
    plt.plot(generations, best_without, 'r-', linewidth=2, label='Without Elitism')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.title('Best Fitness: With vs Without Elitism')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 5)

    plt.subplot(1, 2, 2)
    plt.plot(generations, avg_with, 'b-', linewidth=2, label='With Elitism')
    plt.plot(generations, avg_without, 'r-', linewidth=2, label='Without Elitism')
    plt.xlabel('Generation')
    plt.ylabel('Average Fitness')
    plt.title('Average Fitness: With vs Without Elitism')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 5)
    
    plt.tight_layout()
    plt.savefig('elitism_comparison.png')
    plt.show()
    



if __name__ == "__main__":
  
    bestHistories, avgHistories = runMultipleRuns(5)
    printFinalPopulation()
    
    plotElitismComparison()
