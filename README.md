# NE-AMNC
## Usage Example
### NE-AMNC
Cora 
```javascript 
python main.py --dataset Cora --runs 10 --epochs 1300 --dropout 0.1 --hidden 300 --hidden_z 300 --early_stopping 20 --lr=0.005 --weight_decay 0.005 --alph 2 --beta 3 --K 3 --augmentation MIP --layer 1 2 3
```
Pubmed
```javascript 
python main.py --dataset Pubmed --runs 10 --epochs 800 --dropout 0.0 --hidden 400 --hidden_z 400 --early_stopping 20 --lr=0.01 --weight_decay 0.0005 --alph 2 --beta 3 --K 3 --augmentation MIP --layer 1 2 3  --multi_layer 10
```
Wisconsin
```javascript 
python main.py --dataset Wisconsin --runs 10 --epochs 600 --dropout 0.1 --hidden 300 --hidden_z 300 --early_stopping 10 --lr=0.006 --weight_decay 0 --alph 0.05 --beta 0.06 --K 3 --augmentation MIP --layer 4 5  --multi_layer 10
```
Actor
```javascript 
python main.py --dataset Actor --runs 10 --epochs 500 --dropout 0 --normalize_features True --hidden 512 --hidden_z 512 --early_stopping 10 --lr=0.02 --weight_decay 0.003 --alph 2 --beta 3 --K 2 --augmentation MIP --layer 1 2  --multi_layer 10
```
## Results
model	|Cora	|CiteSeer	|PubMed|Cornell|Texas	|Wisconsin	|Actor
------ | -----  |----------- |---|--- | -----  |----------- |-------
NE-AMNC|	88.1% |	76.8%|	90.2%|86.4%|	85.7% |	91.7%|41.6%

