"""
Artificial Butterfly Optimization (ABO) Algorithm for Feature Selection
=======================================================================
Inspired by the foraging and mating behaviors of butterflies:
- Sunspot Flying Mode: Butterflies fly toward the best known position (sunspot)
- Canopy Flight Mode: Exploration around the current position
- Free Flight Mode: Random exploration to avoid local optima
- Male Competition: Males compete for the best sunspot positions
"""

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier


class ArtificialButterflyOptimizer:
    """
    Artificial Butterfly Optimizer for binary feature selection.
    
    Each butterfly represents a binary feature mask where 1 = selected, 0 = not selected.
    The fitness function evaluates the classification accuracy using only the selected features.
    """
    
    def __init__(self, n_butterflies=30, max_iter=50, p_sunspot=0.8, p_canopy=0.6,
                 crossover_rate=0.5, mutation_rate=0.1, min_features=5, random_state=42):
        """
        Parameters
        ----------
        n_butterflies : int
            Population size (number of butterflies)
        max_iter : int
            Maximum number of iterations
        p_sunspot : float
            Probability of sunspot flying mode (exploitation)
        p_canopy : float
            Probability of canopy flight mode (within sunspot mode)
        crossover_rate : float
            Crossover rate for male competition
        mutation_rate : float
            Mutation rate for maintaining diversity
        min_features : int
            Minimum number of features to select
        random_state : int
            Random seed for reproducibility
        """
        self.n_butterflies = n_butterflies
        self.max_iter = max_iter
        self.p_sunspot = p_sunspot
        self.p_canopy = p_canopy
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.min_features = min_features
        self.random_state = random_state
        
        self.best_position = None
        self.best_fitness = -np.inf
        self.best_features = None
        self.convergence_history = []
        
    def _initialize_population(self, n_features):
        """Initialize butterfly population with random binary positions."""
        rng = np.random.RandomState(self.random_state)
        population = rng.randint(0, 2, size=(self.n_butterflies, n_features))
        
        # Ensure each butterfly selects at least min_features
        for i in range(self.n_butterflies):
            if np.sum(population[i]) < self.min_features:
                indices = rng.choice(n_features, self.min_features, replace=False)
                population[i][indices] = 1
                
        return population
    
    def _fitness(self, position, X, y, classifier):
        """
        Evaluate fitness of a feature subset.
        
        Fitness = alpha * accuracy + (1 - alpha) * (1 - |selected|/|total|)
        This balances accuracy with feature reduction.
        """
        selected = np.where(position == 1)[0]
        
        if len(selected) < self.min_features:
            return -1.0
        
        X_selected = X[:, selected]
        
        try:
            scores = cross_val_score(classifier, X_selected, y, cv=3, scoring='accuracy', n_jobs=-1)
            accuracy = np.mean(scores)
        except Exception:
            return -1.0
        
        # Feature reduction ratio (reward fewer features)
        alpha = 0.9
        reduction = 1.0 - (len(selected) / X.shape[1])
        fitness = alpha * accuracy + (1 - alpha) * reduction
        
        return fitness
    
    def _sunspot_flying(self, butterfly, best_pos, rng):
        """
        Sunspot Flying Mode: Move toward the best known position.
        Butterflies are attracted to the sunspot (best solution).
        """
        new_pos = butterfly.copy()
        n_features = len(butterfly)
        
        # Canopy flight (local search around best position)
        if rng.random() < self.p_canopy:
            # Follow the best position with perturbation
            for j in range(n_features):
                if rng.random() < 0.7:
                    new_pos[j] = best_pos[j]
                else:
                    # Small perturbation
                    new_pos[j] = 1 - new_pos[j]
        else:
            # Direct flight toward sunspot
            step = rng.random(n_features)
            for j in range(n_features):
                if step[j] > 0.5:
                    new_pos[j] = best_pos[j]
                    
        return new_pos
    
    def _free_flight(self, butterfly, rng):
        """
        Free Flight Mode: Random exploration to escape local optima.
        Butterflies fly randomly to explore new regions of the search space.
        """
        new_pos = butterfly.copy()
        n_features = len(butterfly)
        
        # Levy flight-inspired random walk
        n_flip = max(1, int(rng.exponential(scale=3)))
        flip_indices = rng.choice(n_features, min(n_flip, n_features), replace=False)
        new_pos[flip_indices] = 1 - new_pos[flip_indices]
        
        return new_pos
    
    def _male_competition(self, pop, fitness_vals, rng):
        """
        Male Butterfly Competition: Males compete for the best sunspot positions.
        Top males produce offspring through crossover, replacing weaker individuals.
        """
        n = len(pop)
        sorted_indices = np.argsort(fitness_vals)[::-1]  # Best first
        
        # Top 30% are winners, bottom 30% are replaced
        n_winners = max(2, n // 3)
        n_losers = n // 3
        
        winners = sorted_indices[:n_winners]
        losers = sorted_indices[-n_losers:]
        
        for i, loser_idx in enumerate(losers):
            # Select two winners for crossover
            parent1 = pop[winners[rng.randint(0, n_winners)]]
            parent2 = pop[winners[rng.randint(0, n_winners)]]
            
            # Uniform crossover
            child = parent1.copy()
            mask = rng.random(len(child)) < self.crossover_rate
            child[mask] = parent2[mask]
            
            # Mutation
            mut_mask = rng.random(len(child)) < self.mutation_rate
            child[mut_mask] = 1 - child[mut_mask]
            
            pop[loser_idx] = child
            
        return pop
    
    def _ensure_min_features(self, position, rng):
        """Ensure the butterfly selects at least min_features."""
        if np.sum(position) < self.min_features:
            zero_indices = np.where(position == 0)[0]
            n_needed = self.min_features - int(np.sum(position))
            if len(zero_indices) >= n_needed:
                activate = rng.choice(zero_indices, n_needed, replace=False)
                position[activate] = 1
        return position
    
    def optimize(self, X, y, classifier=None, verbose=True):
        """
        Run the Artificial Butterfly Optimization algorithm.
        
        Parameters
        ----------
        X : numpy array
            Feature matrix (n_samples, n_features)
        y : numpy array
            Target labels
        classifier : sklearn estimator
            Classifier to evaluate feature subsets (default: DecisionTree)
        verbose : bool
            Print progress
            
        Returns
        -------
        selected_features : numpy array
            Indices of selected features
        best_fitness : float
            Best fitness achieved
        """
        if classifier is None:
            classifier = DecisionTreeClassifier(random_state=self.random_state, max_depth=10)
        
        rng = np.random.RandomState(self.random_state)
        n_features = X.shape[1]
        
        # Initialize population
        population = self._initialize_population(n_features)
        fitness_vals = np.zeros(self.n_butterflies)
        
        if verbose:
            print(f"{'='*60}")
            print(f"Artificial Butterfly Optimizer — Feature Selection")
            print(f"{'='*60}")
            print(f"Population: {self.n_butterflies} | Iterations: {self.max_iter}")
            print(f"Features: {n_features} | Min Features: {self.min_features}")
            print(f"{'='*60}")
        
        # Evaluate initial population
        for i in range(self.n_butterflies):
            fitness_vals[i] = self._fitness(population[i], X, y, classifier)
            
        # Track global best
        best_idx = np.argmax(fitness_vals)
        self.best_position = population[best_idx].copy()
        self.best_fitness = fitness_vals[best_idx]
        
        for iteration in range(self.max_iter):
            new_population = population.copy()
            
            for i in range(self.n_butterflies):
                if rng.random() < self.p_sunspot:
                    # Sunspot Flying Mode (exploitation)
                    new_population[i] = self._sunspot_flying(
                        population[i], self.best_position, rng
                    )
                else:
                    # Free Flight Mode (exploration)
                    new_population[i] = self._free_flight(population[i], rng)
                
                # Ensure minimum features
                new_population[i] = self._ensure_min_features(new_population[i], rng)
                
                # Evaluate new position
                new_fitness = self._fitness(new_population[i], X, y, classifier)
                
                # Greedy selection
                if new_fitness > fitness_vals[i]:
                    population[i] = new_population[i]
                    fitness_vals[i] = new_fitness
                    
                    # Update global best
                    if new_fitness > self.best_fitness:
                        self.best_fitness = new_fitness
                        self.best_position = new_population[i].copy()
            
            # Male competition phase (every 5 iterations)
            if (iteration + 1) % 5 == 0:
                population = self._male_competition(population, fitness_vals, rng)
                # Re-evaluate after competition
                for i in range(self.n_butterflies):
                    population[i] = self._ensure_min_features(population[i], rng)
                    fitness_vals[i] = self._fitness(population[i], X, y, classifier)
                    if fitness_vals[i] > self.best_fitness:
                        self.best_fitness = fitness_vals[i]
                        self.best_position = population[i].copy()
            
            self.convergence_history.append(self.best_fitness)
            
            if verbose and (iteration + 1) % 10 == 0:
                n_selected = int(np.sum(self.best_position))
                print(f"  Iteration {iteration+1:3d}/{self.max_iter} | "
                      f"Best Fitness: {self.best_fitness:.4f} | "
                      f"Features Selected: {n_selected}/{n_features}")
        
        # Get selected feature indices
        self.best_features = np.where(self.best_position == 1)[0]
        
        if verbose:
            print(f"{'='*60}")
            print(f"Optimization Complete!")
            print(f"  Best Fitness: {self.best_fitness:.4f}")
            print(f"  Features Selected: {len(self.best_features)}/{n_features}")
            print(f"  Selected Indices: {self.best_features}")
            print(f"{'='*60}")
        
        return self.best_features, self.best_fitness
    
    def get_convergence_history(self):
        """Return the convergence history for plotting."""
        return self.convergence_history
    
    def transform(self, X):
        """Apply feature selection to new data."""
        if self.best_features is None:
            raise ValueError("Optimizer has not been run yet. Call optimize() first.")
        return X[:, self.best_features]


if __name__ == "__main__":
    # Quick test
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=500, n_features=30, n_informative=10,
                                n_redundant=10, random_state=42)
    
    abo = ArtificialButterflyOptimizer(n_butterflies=20, max_iter=30, random_state=42)
    selected, fitness = abo.optimize(X, y)
    print(f"\nSelected {len(selected)} features: {selected}")
