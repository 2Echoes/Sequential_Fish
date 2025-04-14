"""
Submodule for co-localization analysis

# Notation

We try co-localization between 2 populations *population1* --> *population2*, meaning we count how many single from population 1 
co-localize with population 2.
Those populations can be selected from 'all', 'free' or 'clustered' spots.  
    
V is volume in pixel by which we mean the number of available position a single molecule can take in a cell
    
# Notes

- **Volume** :  
    In analysis we can't acess volume of a cell with precision as cell are segmented in 2D, computing volume by multiplying 
Area with z-stack number would consider a lot of pixel that single molecules can't acess in reality. To work around this we compute first
the average number of single molecule per plane and work with modelisation in 2D as our model is dimensionly independent. For each cell
number of single molecule per plane and 'volume' of plane (pixels) then we compute expected colocalization using  
**'plane abudancy'** = volume x spot per plane.

- **Statistical test** :  
    With our modelisation we have acces to probabilistic densities of colocalization events for each cell. Though for each cell 
    (i.e particular set of volume and abudancies) we only have one measure of co-colocalization event meaning we can't statisticaly test
    the relevance of this measurement, anyway it would not be interesting as **the normalised co-localisation rate** already
    is a indication of how far we are from random distribution. When then average these events measurement on the whole cell population,
    the statistical test is to know if this average value (also average normalised valued) is statistically significant compared to what
    we would expect from a set of cells with randomly distributed single molecules. To do so we need to compute, in a way,
    the distribution of distributions of co-localization events for a bach of cells (i.e a set of volume and abundancies).
    
    Hopefully, **the distribution of random variables following normal law also follow normal law** and we can compute its mean and std.
    (from propertis of gaussan and linear combinations)
    Mean is the mean of each individual distribution (knowing volume and abudancies) (law of total expectation)
    Std is the sum of squared variance(or std ???) divided by N squared.
    
    From this new normal distribution we can use usual statistical tests.

"""
import pandas as pd
import numpy as np

from typing import Literal

from .models import compute_colocalization_count_expectancy, compute_colocalization_count_std, compute_colocalization_probability
from .models import compute_unique_position_expectancy
from .models import compute_unique_pair_expectancy, compute_unique_pair_std

from ..tools import safe_merge_no_duplicates

from sklearn.neighbors import NearestNeighbors


"""
1. Data co-localization pariwise rates : Compute actual co-colocalization pairwise rates.
"""

def _get_population_index(
    Spots : pd.DataFrame, 
    population_key:Literal['all','free','clustered']
    ) :
    """
    Get index from Spots data frame to a population : 'all', 'clustered', 'free'
    """
    if population_key == 'all' : population_index = Spots.index
    elif population_key == 'clustered' : population_index = Spots.loc[~Spots['cluster_id'].isna()].index
    elif population_key == 'cluster' : population_index = Spots.loc[~Spots['cluster_id'].isna()].index
    elif population_key == 'clusters' : population_index = Spots.loc[~Spots['cluster_id'].isna()].index
    elif population_key == 'free' : population_index = Spots.loc[Spots['cluster_id'].isna()].index
    else : raise AssertionError("{} incorect key for population_key".format(population_key))

    return population_index

def _create_coordinate_df(
    Spots : pd.DataFrame, 
    population_key:Literal['all','free','clustered']
    ) :
    """
    Prepare dataframe with nanometer coordinates
    """
    
    population_index = _get_population_index(Spots, population_key)
    coordinates_df = Spots.loc[population_index]

    #converting pixel coordinates to nanometers
    coordinates_df['voxel_size_z'], coordinates_df['voxel_size_y'], coordinates_df['voxel_size_x'] = list(zip(*coordinates_df['voxel_size']))
    coordinates_df['z'] *= coordinates_df['voxel_size_z'] 
    coordinates_df['y'] *= coordinates_df['voxel_size_y'] 
    coordinates_df['x'] *= coordinates_df['voxel_size_x'] 

    coordinates_df['coordinates'] = list(zip(coordinates_df['z'], coordinates_df['y'], coordinates_df['x']))
    coordinates_df = coordinates_df.groupby(['location','target'])['coordinates'].apply(list)
    return coordinates_df

def _create_neighbor_model_dict(
    spots_coordinates_df : pd.DataFrame, 
    colocalisation_distance : int
    ) :
    """
    Prepare for each single distribution a Nearestneighbor model, population passed to those must be 'population2' in other words the population where co-localization is tried WITH. See submodule description above.
    """
    neighbor_models_dict = dict()
    for idx in spots_coordinates_df.index :
        spot_distribution = spots_coordinates_df.at[idx]
        new_model = NearestNeighbors(radius=colocalisation_distance)
        new_model.fit(spot_distribution)
        neighbor_models_dict[idx] = new_model
        
    return neighbor_models_dict

def _compute_colocalisation_truth_df(
    spots_coordinates_df : pd.DataFrame, 
    Spots : pd.DataFrame, 
    neighbor_models_dict : dict,
    population_1 : Literal['all','clustered','free'],
    ) :
    
    population1_index = _get_population_index(Spots, population_key=population_1)
    RNAs = list(spots_coordinates_df.index.get_level_values(1).unique())
    colocalisation_truth_df = pd.DataFrame(index=population1_index, columns= RNAs, dtype=bool)
    colocalisation_truth_df = colocalisation_truth_df.join(Spots.loc[:,['spot_id','location','target', 'z','y','x','voxel_size']])

    #converting coordinates to nanometers
    colocalisation_truth_df['voxel_size_z'], colocalisation_truth_df['voxel_size_y'], colocalisation_truth_df['voxel_size_x'] = list(zip(*colocalisation_truth_df['voxel_size']))
    colocalisation_truth_df['z'] *= colocalisation_truth_df['voxel_size_z'] 
    colocalisation_truth_df['y'] *= colocalisation_truth_df['voxel_size_y'] 
    colocalisation_truth_df['x'] *= colocalisation_truth_df['voxel_size_x'] 
    colocalisation_truth_df['coordinates'] = list(zip(colocalisation_truth_df['z'], colocalisation_truth_df['y'], colocalisation_truth_df['x']))

    colocalisation_truth_df = colocalisation_truth_df.drop(columns=['z','y','x','voxel_size','voxel_size_z','voxel_size_y','voxel_size_x'])
    
    for location in colocalisation_truth_df['location'].unique() :
        target_idx = colocalisation_truth_df[colocalisation_truth_df['location'] == location].index
        for rna in RNAs :
            model : NearestNeighbors = neighbor_models_dict[(location, rna)]
            coordinates = list(colocalisation_truth_df.loc[target_idx]['coordinates'].apply(np.array,dtype=int))
            coordinates = np.array(coordinates, dtype=int)
            query = model.radius_neighbors(coordinates, return_distance=False)
            query = pd.Series(query, index=target_idx).apply(len).apply(bool) #if count is 0 no colocalisation -> False else True
            colocalisation_truth_df.loc[target_idx,[rna]] = query
    
    return colocalisation_truth_df

def _spots_merge_data(
    Spots : pd.DataFrame,
    Acquisition : pd.DataFrame,
    Detection : pd.DataFrame,
    Gene_map : pd.DataFrame,
    ) :
    """
    Merge required information into Spots df
    """
    
    Detection = safe_merge_no_duplicates(
    Detection,
    Acquisition,
    on= ['acquisition_id'],
    keys=['cycle','location', 'fish_reodered_shape']
)

    Detection = safe_merge_no_duplicates(
        Detection,
        Gene_map,
        on= ['cycle','color_id'],
        keys=['target']
    )

    Spots =safe_merge_no_duplicates(
        Spots,
        Detection,
        on= 'detection_id',
        keys= ['location','target', 'voxel_size', 'fish_reodered_shape']
    )


    return Spots

def colocalisation_truth_df(
    Spots : pd.DataFrame,
    Acquisition : pd.DataFrame,
    Detection : pd.DataFrame,
    Gene_map : pd.DataFrame,
    population_1 : Literal['all','clustered','free'] = 'all',
    population_2 : Literal['all','clustered','free']= 'all',
    colocalisation_distance : int = 400,
    ) :
    """
    
    Create a dataframe where each line corresponds to one spot

    PARAMETERS
    ----------
        Spots must contain voxel_size.
    
    KEYS
    ----
        - `location` : field of fiew reference
        - `target` : from which distribution comes this spot
        - `coordinates` : ...
        - `spot_id` : unique identifier to spot
        
        - `boolean key` : + one key for each different `target` value, representing a boolean value indicating TRUE if this spot co-localize with this distribution
    
    """
    Spots = _spots_merge_data(
        Spots=Spots,
        Detection=Detection,
        Acquisition=Acquisition,
        Gene_map=Gene_map
    )
    
    population_1_index = _get_population_index(
        Spots, 
        population_key=population_1
        )
    
    real_spots_coordinates_df = _create_coordinate_df(
        Spots, 
        population_key= population_2
        )
    
    neighbor_models_dict = _create_neighbor_model_dict(
        real_spots_coordinates_df, 
        colocalisation_distance=colocalisation_distance
        )
    
    colocalisation_truth_df = _compute_colocalisation_truth_df(
        real_spots_coordinates_df, 
        Spots.loc[population_1_index], 
        neighbor_models_dict,
        population_1=population_1,
        )
    
    return colocalisation_truth_df

def create_cell_coloc_rates_df(
    Spots : pd.DataFrame, 
    colocalisation_truth_df : pd.DataFrame,
    ) -> pd.DataFrame:
    """
    Create dataframe where each index and columns correspond to `(target, cell_id)` values (i.e one line per distributions, cell_id couple)
    
    Colocalization rate of target i with target j is found at line i, column j (i.e `df.at[i,j]`)
    
    Result is ready for mean calculation or normalisation.
    
    """
    RNA_list = list(Spots['target'].unique())
    RNA_list.sort()
    
    colocalisation_truth_df= safe_merge_no_duplicates(
        colocalisation_truth_df,
        Spots,
        on='spot_id',
        keys='cell_id'
    )

    cell_coloc_rates = colocalisation_truth_df.groupby(['target','cell_id'])[RNA_list].mean() #Normalisation needs to happen after this


    return cell_coloc_rates

def compute_coloc_rates_mean(
    cell_coloc_rates : pd.DataFrame,
    ) :
    
    """
    Return dataframe with one line per distribution with mean value (amongst cell population) of co-localization.
    Colocalization rate of target i with target j is found at line i, column j (i.e `df.at[i,j]`)
    """
    
    coloc_rates = cell_coloc_rates.groupby('target', axis=0, level=0).mean()
    return coloc_rates
    

"""
2. Co-localization normalisation : Computes co-localization scores from modelisation

Aim : normalize cell_coloc_rates from part 1. with modelisation expectancy.
"""

def _compute_corrected_volume(
    voxel_size : 'tuple[int]',
    colocalisation_distance : int,
    Cell_area : pd.DataFrame,
    ) :
    pass #TODO

def _get_cell_area(
    Cell : pd.DataFrame,
    ) :
    assert (Cell.groupby('cell_id',as_index=True)['cell_area'].unique().apply(len) == 1).all(), "Cell area is not unique for cell_id"
    Cell_area = Cell.groupby('cell_id',as_index=True)['cell_area'].first() #is unique so we take first

    return Cell_area

def _get_spot_per_plane(
    Spots : pd.DataFrame,
    ) :
    Cell_spots_count : pd.DataFrame = Spots.groupby(['cell_id','target','z'], as_index=False)['spot_id'].count()
    Cell_spots_count : pd.DataFrame = Cell_spots_count.groupby(['cell_id','target'], as_index=False)['spot_id'].mean().rename(columns={'spot_id' : 'spot_per_plane'})
    
    return Cell_spots_count

def _compute_spot_density(
    spots_per_plane : pd.DataFrame,
    Cell_area : pd.Series,
    RNA_list : 'list[str]',
    ):
    """
    For each distribution (columns) compute density of single molecule per plane, one line per cell_id.
    """

    Cell_spot_density = spots_per_plane.pivot(columns='target',index='cell_id',values='spot_per_plane')
    for rna in RNA_list :
        Cell_spot_density[rna] = spots_per_plane[rna]/Cell_area
    
    return Cell_spot_density
   
def _compute_cell_coloc_rate_expectancy(
    Cell_spot_density : pd.DataFrame,
    Cell_area : pd.DataFrame,
    ):   
    
    check_len = len(Cell_spot_density)
    Cell_df = pd.merge(
        Cell_spot_density.reset_index(level=0, drop=False), #is on axis 0 level 1 for left and axis 0 level 0 for right so reset put target back in columns
        Cell_area,
        on='cell_id', 
        left_index=True,
        right_index=True
    )
    assert len(Cell_df) == check_len, "error in line conservation when merging cell area with cell density on cell_id"
    
    Cell_df['equivalent_abundancy'] = Cell_df['cell_area'] * Cell_df['spot_per_plane']
    
    coloc_expectancy = Cell_df.apply(
        lambda x: compute_colocalization_probability(
            V= x['cell_area'],
            a1= x['equivalent_abundancy']
        ), axis= 1 # strangely for axis=0 x is column and for axis = 1 x is lines
        )
        
    return pd.DataFrame(coloc_expectancy.rename('cell_coloc_rate_expectancy'))

def _compute_coloc_rate_expectancy(
    cell_coloc_expectancy : pd.DataFrame,
) :
    
    return cell_coloc_expectancy.groupby('target').mean()

def create_coloc_rate_expectancy(
    Spots : pd.DataFrame,
    Cell : pd.DataFrame,
    voxel_size : 'tuple[int]',
    colocalisation_distance : int
) :
    """
    Return dataframe with one line per distribution with mean value (amongst cell population) of co-localization rate expectancies.
    Colocalization rate of target i with target j is found at line i, column j (i.e `df.at[i,j]`)
    """
    
    Cell_area = _get_cell_area(Cell)
    Cell_area = _compute_corrected_volume(
        voxel_size=voxel_size,
        colocalisation_distance=colocalisation_distance,
        Cell_area=Cell_area
    )
    
    RNA_list = list(Spots['target'].unique())
    
    Spots_per_plane = _get_spot_per_plane(Spots)
    Spots_density = _compute_spot_density(
        spots_per_plane=Spots_per_plane,
        Cell_area=Cell_area,
        RNA_list= RNA_list,
    )
    
    coloc_rate_expectancy = _compute_cell_coloc_rate_expectancy(
        Cell_spot_density=Spots_density,
        Cell_area=Cell_area,
    )
    
    coloc_rate_expectancy = _compute_coloc_rate_expectancy(
        cell_coloc_expectancy=coloc_rate_expectancy
    )
    
    return coloc_rate_expectancy

def normalise_coloc_rate(
    coloc_rates : pd.DataFrame,
    coloc_rate_expectancy : pd.DataFrame,
) :
    
    assert coloc_rates.index.equals(coloc_rate_expectancy.index), "Index doesn't match between experimental coloc rates and model coloc rates"
    assert coloc_rates.columns.equals(coloc_rate_expectancy.columns), "Columns Index doesn't match between experimental coloc rates and model coloc rates"

    coloc_rate_expectancy = coloc_rate_expectancy.replace(0, np.NaN) #Avoid division by 0 error
    
    return coloc_rates / coloc_rate_expectancy
    
    
"""
3. Statistical test (p-value) VS Null-model

We already created function for mean computation in a distribution -> `_compute_coloc_rate_expectancy`. But we want the equivalent with standard deviation.

"""

def _compute_cell_distribution_populations(
        Spots : pd.DataFrame,
) :
    Cell_spots_count : pd.DataFrame = Spots.groupby(['cell_id','target'], as_index=False)['spot_id'].count().rename('abundancy')

    return Cell_spots_count

def compute_cell_distribution_std(
    Cell_spot_density : pd.DataFrame,
    Cell_area : pd.DataFrame,
    Cell_distribution_populations : pd.DataFrame,
    ):
    """
    Compute expected standard deviation for cells (for each cell computes std according to its Volume and abundancies)
    """

    
    check_len = len(Cell_spot_density)
    Cell_df = pd.merge(
        Cell_spot_density.reset_index(level=0, drop=False), #cell_id is on axis 0 level 1 for left and axis 0 level 0 for right so reset put target back in columns
        Cell_area,
        on='cell_id', 
        left_index=True,
        right_index=True
    )
    assert len(Cell_df) == check_len, "error in line conservation when merging cell area with cell density on cell_id"
    Cell_df['a1_equivalent_abundancy'] = Cell_df['cell_area'] * Cell_df['spot_per_plane']
    
    Cell_distribution_populations = Cell_distribution_populations.reset_index(drop=True, level=0) #Drop target info as it is in Cell_df already

    Cell_df = pd.merge(
        Cell_df,
        Cell_distribution_populations,
        on='cell_id',
        left_index=True,
        right_index=True,
    )

    cell_coloc_standard_deviation = Cell_df.apply(
        lambda x: compute_colocalization_count_std(
            a1_unique= x['equivalent_abundancy'],
            a2= x['abundancy'],
            V= x['cell_area'],
        ), axis= 1 # strangely for axis=0 x is column and for axis = 1 x is lines
        )

    return pd.DataFrame(cell_coloc_standard_deviation.rename("cell_coloc_events_standard_deviation"))

def compute_distributions_models(
        cell_coloc_expectancy : pd.DataFrame
) : 
    pass

"""
4. Higher dimension co-localization tests
"""