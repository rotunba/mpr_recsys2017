package librec.ranking;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import librec.data.Configuration;
import librec.data.DenseMatrix;
import librec.data.MatrixEntry;
import librec.data.SparseMatrix;
import librec.intf.IterativeRecommender;
import librec.util.LineConfiger;

@Configuration("binThold, alpha, factors, regU, regI, numIters")
public class eALS extends IterativeRecommender {
	protected float weightCoefficient;
	private float ratio;
	private float overallWeight;
	private int WRMFJudge;
	private double[] confidences;
	private SparseMatrix weights;

	public eALS(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
		super(trainMatrix, testMatrix, fold);
	}

	protected void initModel() throws Exception {
		super.initModel();
		this.weightCoefficient = algoOptions.getFloat("-rec.wrmf.weight.coefficient", 4.0F);
		this.ratio = algoOptions.getFloat("rec.eals.ratio", 0.4F);
		this.overallWeight = algoOptions.getFloat("rec.eals.overall", 128.0F);
		this.WRMFJudge = algoOptions.getInt("rec.eals.wrmf.judge", 1);

		this.confidences = new double[numItems];
		this.weights = new SparseMatrix(this.trainMatrix);

		initConfidencesAndWeights();
	}

	private void initConfidencesAndWeights() {
		if ((this.WRMFJudge == 0) || (this.WRMFJudge == 2)) {
			double sumPopularity = 0.0D;

			for (int itemIdx = 0; itemIdx < numItems; itemIdx++) {
				double alphaPopularity = Math.pow(this.trainMatrix.columnSize(itemIdx) * 1.0D / numRates, this.ratio);
				this.confidences[itemIdx] = (this.overallWeight * alphaPopularity);
				sumPopularity += alphaPopularity;
			}
			for (int itemIdx = 0; itemIdx < numItems; itemIdx++)
				this.confidences[itemIdx] /= sumPopularity;
		} else {
			for (int itemIdx = 0; itemIdx < numItems; itemIdx++) {
				this.confidences[itemIdx] = 1.0D;
			}

		}

		for (MatrixEntry matrixEntry : this.trainMatrix) {
			int userIdx = matrixEntry.row();
			int itemIdx = matrixEntry.column();
			if ((this.WRMFJudge == 1) || (this.WRMFJudge == 2)) {
				this.weights.set(userIdx, itemIdx, 1.0D + this.weightCoefficient * matrixEntry.get());
			} else {
				this.weights.set(userIdx, itemIdx, 1.0D);
			}
		}
	}

	protected void buildModel() throws Exception {
		List userItemsList = getUserItemsList(this.trainMatrix);
		List itemUsersList = getItemUsersList(this.trainMatrix);

		double[] usersPredictions = new double[numUsers];
		double[] itemsPredictions = new double[numItems];
		double[] usersWeights = new double[numUsers];
		double[] itemsWeights = new double[numItems];

		DenseMatrix itemFactorsCache = new DenseMatrix(numFactors, numFactors);

		for (int iter = 1; iter <= numIters; iter++) {
			double value;
			for (int factorIdx1 = 0; factorIdx1 < numFactors; factorIdx1++)
				for (int factorIdx2 = 0; factorIdx2 <= factorIdx1; factorIdx2++) {
					value = 0.0D;
					for (int itemIdx = 0; itemIdx < numItems; itemIdx++) {
						value += this.confidences[itemIdx] * this.Q.get(itemIdx, factorIdx1)
								* this.Q.get(itemIdx, factorIdx2);
					}
					itemFactorsCache.set(factorIdx1, factorIdx2, value);
					itemFactorsCache.set(factorIdx2, factorIdx1, value);
				}
			double numer;
			Iterator localIterator;
			for (int userIdx = 0; userIdx < numUsers; userIdx++) {
				for (localIterator = ((List) userItemsList.get(userIdx)).iterator(); localIterator.hasNext();) {
					int itemIdx = ((Integer) localIterator.next()).intValue();
					itemsPredictions[itemIdx] = DenseMatrix.rowMult(this.P, userIdx, this.Q, itemIdx);
					itemsWeights[itemIdx] = this.weights.get(userIdx, itemIdx);
				}

				for (int factorCacheIdx = 0; factorCacheIdx < numFactors; factorCacheIdx++) {
					numer = 0.0D;
					double denom = regU + itemFactorsCache.get(factorCacheIdx, factorCacheIdx);

					for (int factorIdx = 0; factorIdx < numFactors; factorIdx++) {
						if (factorCacheIdx != factorIdx) {
							numer -= this.P.get(userIdx, factorIdx) * itemFactorsCache.get(factorCacheIdx, factorIdx);
						}
					}

					for (localIterator = ((List) userItemsList.get(userIdx)).iterator(); localIterator.hasNext();) {
						int itemIdx = ((Integer) localIterator.next()).intValue();
						itemsPredictions[itemIdx] -= this.P.get(userIdx, factorCacheIdx)
								* this.Q.get(itemIdx, factorCacheIdx);

						numer = numer + (itemsWeights[itemIdx]
								- (itemsWeights[itemIdx] - this.confidences[itemIdx]) * itemsPredictions[itemIdx])
								* this.Q.get(itemIdx, factorCacheIdx);

						denom = denom + (itemsWeights[itemIdx] - this.confidences[itemIdx])
								* this.Q.get(itemIdx, factorCacheIdx) * this.Q.get(itemIdx, factorCacheIdx);
					}

					this.P.set(userIdx, factorCacheIdx, numer / denom);
					for (localIterator = ((List) userItemsList.get(userIdx)).iterator(); localIterator.hasNext();) {
						int itemIdx = ((Integer) localIterator.next()).intValue();
						itemsPredictions[itemIdx] += this.P.get(userIdx, factorCacheIdx)
								* this.Q.get(itemIdx, factorCacheIdx);
					}
				}
			}

			DenseMatrix userFactorsCache = this.P.transpose().mult(this.P);

			for (int itemIdx = 0; itemIdx < numItems; itemIdx++) {
				for (localIterator = ((List) itemUsersList.get(itemIdx)).iterator(); localIterator.hasNext();) {
					int userIdx = ((Integer) localIterator.next()).intValue();
					usersPredictions[userIdx] = DenseMatrix.rowMult(this.P, userIdx, this.Q, itemIdx);
					usersWeights[userIdx] = this.weights.get(userIdx, itemIdx);
				}

				for (int factorCacheIdx = 0; factorCacheIdx < numFactors; factorCacheIdx++) {
					 numer = 0.0D;
					double denom = this.confidences[itemIdx] * userFactorsCache.get(factorCacheIdx, factorCacheIdx)
							+ regI;

					for (int factorIdx = 0; factorIdx < numFactors; factorIdx++) {
						if (factorCacheIdx != factorIdx) {
							numer -= this.Q.get(itemIdx, factorIdx) * userFactorsCache.get(factorIdx, factorCacheIdx);
						}
					}
					numer *= this.confidences[itemIdx];

					for (localIterator = ((List) itemUsersList.get(itemIdx)).iterator(); localIterator.hasNext();) {
						int userIdx = ((Integer) localIterator.next()).intValue();
						usersPredictions[userIdx] -= this.P.get(userIdx, factorCacheIdx)
								* this.Q.get(itemIdx, factorCacheIdx);

						numer = numer + (usersWeights[userIdx]
								- (usersWeights[userIdx] - this.confidences[itemIdx]) * usersPredictions[userIdx])
								* this.P.get(userIdx, factorCacheIdx);

						denom = denom + (usersWeights[userIdx] - this.confidences[itemIdx])
								* this.P.get(userIdx, factorCacheIdx) * this.P.get(userIdx, factorCacheIdx);
					}

					this.Q.set(itemIdx, factorCacheIdx, numer / denom);
					for (localIterator = ((List) itemUsersList.get(itemIdx)).iterator(); localIterator.hasNext();) {
						int userIdx = ((Integer) localIterator.next()).intValue();
						usersPredictions[userIdx] += this.P.get(userIdx, factorCacheIdx)
								* this.Q.get(itemIdx, factorCacheIdx);
					}
				}
			}
		}
	}

	private List<List<Integer>> getUserItemsList(SparseMatrix sparseMatrix) {
		List userItemsList = new ArrayList();
		for (int userIdx = 0; userIdx < numUsers; userIdx++) {
			userItemsList.add(sparseMatrix.getColumns(userIdx));
		}
		return userItemsList;
	}

	private List<List<Integer>> getItemUsersList(SparseMatrix sparseMatrix) {
		List itemUsersList = new ArrayList();
		for (int itemIdx = 0; itemIdx < numItems; itemIdx++) {
			itemUsersList.add(sparseMatrix.getRows(itemIdx));
		}
		return itemUsersList;
	}
}