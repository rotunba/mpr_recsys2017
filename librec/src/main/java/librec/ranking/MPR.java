package librec.ranking;

import librec.data.SparseMatrix;
import librec.data.SparseVector;
import librec.intf.IterativeRecommender;
import librec.util.Randoms;
import librec.util.Strings;

public class MPR extends IterativeRecommender {
	public MPR(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
		super(trainMatrix, testMatrix, fold);

		this.initByNorm = false;
	}

	protected void initModel() throws Exception {
		super.initModel();
		this.itemCache = this.trainMatrix.columnCache(cacheSpec);
		this.userCache = this.trainMatrix.rowCache(cacheSpec);
	}

	protected void buildModel() throws Exception {

		for (int iter = 1; iter <= numIters; iter++) {
			this.loss = 0.0D;
			int s = 0;

			for (int smax = numUsers * 100; s < smax; s++) {
				int u = 0;
				int i = 0;
				int j = 0;
				int k = 0;
				int v = 0;
				int w = 0;
				
				boolean doesUserLikeNothingOrAll = false, isItemLikedByNooneOrByAll = false;
				
				do {
					u = Randoms.uniform(numUsers); //
					SparseVector itemSet = userCache.get(u);

					doesUserLikeNothingOrAll = itemSet.getCount() == 0 || itemSet.getCount() == numItems;

					if (!doesUserLikeNothingOrAll) {
						int[] itemList = itemSet.getIndex();

						i = itemList[Randoms.uniform(itemList.length)];

						int[] diff = diff(numItems, itemList);

						j = diff[Randoms.uniform(diff.length)];
					}

				} while (doesUserLikeNothingOrAll);

				do {
					k = Randoms.uniform(numItems); //
					SparseVector userSet = itemCache.get(k);

					isItemLikedByNooneOrByAll = userSet.getCount() == 0 || userSet.getCount() == numUsers;

					if (!isItemLikedByNooneOrByAll) {
						int[] userList = userSet.getIndex();

						v = userList[Randoms.uniform(userList.length)];

						int[] diff = diff(numUsers, userList);

						w = diff[Randoms.uniform(diff.length)];
					}

				} while (isItemLikedByNooneOrByAll);

				
				double xkv = predict(v, k);
				double xkw = predict(w, k);
				double xui = predict(u, i);
				double xuj = predict(u, j);
				double ykvw = xkv - xkw;
				double xuij = xui - xuj;

				this.loss += Math.log(1.0D + Math.exp(-ykvw)) + Math.log(1.0D + Math.exp(-xuij));

				double cmgxuij = g(-xuij) * 0.5D;
				double cmgxkvw = g(-ykvw) * 0.5D;

				for (int f = 0; f < numFactors; f++) {
					double qkf = this.Q.get(k, f);
					double pvf = this.P.get(v, f);
					double pwf = this.P.get(w, f);

					this.Q.add(k, f, this.lRate * (cmgxkvw * (pvf - pwf) - regI * qkf));
					this.P.add(v, f, this.lRate * (cmgxkvw * qkf - regU * pvf));
					this.P.add(w, f, this.lRate * (cmgxkvw * -qkf - regU * pwf));

					double puf = this.P.get(u, f);
					double qif = this.Q.get(i, f);
					double qjf = this.Q.get(j, f);

					this.P.add(u, f, this.lRate * (cmgxuij * (qif - qjf) - regU * puf));
					this.Q.add(i, f, this.lRate * (cmgxuij * puf - regI * qif));
					this.Q.add(j, f, this.lRate * (cmgxuij * -puf - regI * qjf));

					this.loss += regU * puf * puf + regI * qif * qif + regI * qjf * qjf + regI * qkf * qkf
							+ regU * pvf * pvf + regU * pwf * pwf;
					this.loss *= 0.5D;
				}

			}

			if (isConverged(iter))
				break;
		}
	}

	public String toString() {
		return Strings.toString(
				new Object[] { Float.valueOf(binThold), Integer.valueOf(numFactors), Float.valueOf(initLRate),
						Float.valueOf(maxLRate), Float.valueOf(regU), Float.valueOf(regI), Integer.valueOf(numIters) },
				",");
	}
}