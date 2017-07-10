/**
 * Copyright (C) 2016 LibRec
 * <p>
 * This file is part of LibRec.
 * LibRec is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * <p>
 * LibRec is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 * <p>
 * You should have received a copy of the GNU General Public License
 * along with LibRec. If not, see <http://www.gnu.org/licenses/>.
 */
package net.librec.recommender.cf.ranking;

import net.librec.annotation.ModelData;
import net.librec.common.LibrecException;
import net.librec.math.algorithm.Maths;
import net.librec.math.algorithm.Randoms;
import net.librec.math.structure.MatrixEntry;
import net.librec.math.structure.SparseMatrix;
import net.librec.recommender.MatrixFactorizationRecommender;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Otunba et al., <strong>MPR: Multi-Objective Pairwise Ranking</strong>, RecSys 2017.
 *
 * @author Rasaq Otunba, Raimi A. Rufai and Jessica Lin
 */
@ModelData({"isRanking", "ar", "userFactors", "itemFactors"})
public class ARRecommender extends MatrixFactorizationRecommender {
    private List<Set<Integer>> itemUsersSet;

    @Override
    protected void setup() throws LibrecException {
        super.setup();
    }

    @Override
    protected void trainModel() throws LibrecException {

        itemUsersSet = getItemUsersSet(trainMatrix);

        for (int iter = 1; iter <= numIterations; iter++) {

            loss = 0.0d;
            for (int sampleCount = 0, smax = numItems * 100; sampleCount < smax; sampleCount++) {

                // randomly draw (userIdx, posItemIdx, negItemIdx)
                int itemIdx, posUserIdx = 0, negUserIdx = 0;
               
                boolean isItemLikedByNooneOrByAll = false;
                do {
                	itemIdx = Randoms.uniform(numItems); //
                	Set<Integer> userSet = itemUsersSet.get(itemIdx);
					
                	isItemLikedByNooneOrByAll = userSet.size() == 0 || userSet.size() == numUsers;

					if (!isItemLikedByNooneOrByAll) {
						List<Integer> userList = trainMatrix.getRows(itemIdx);

						posUserIdx = userList.get(Randoms.uniform(userList.size())); // posUserIdx prefers itemIdx

						int[] diff = diff(numUsers, userList); 
						
						negUserIdx = diff[Randoms.uniform(diff.length)]; 
					} 

				} while (isItemLikedByNooneOrByAll);

                // update parameters
                double posPredictRating = predict(posUserIdx, itemIdx);
                double negPredictRating = predict(negUserIdx, itemIdx);
                double diffValue = posPredictRating - negPredictRating;

                double lossValue = Math.log(Maths.logistic(diffValue)); 
                loss += lossValue * 0.5;

                double deriValue = Maths.logistic(-diffValue)  * 0.5;

                for (int factorIdx = 0; factorIdx < numFactors; factorIdx++) {
                    double itemFactorValue = itemFactors.get(itemIdx, factorIdx);
                    double posUserFactorValue = userFactors.get(posUserIdx, factorIdx);
                    double negUserFactorValue = userFactors.get(negUserIdx, factorIdx);

                    itemFactors.add(itemIdx, factorIdx, learnRate * (deriValue * (posUserFactorValue - negUserFactorValue) - regItem * itemFactorValue));
                    userFactors.add(posUserIdx, factorIdx, learnRate * (deriValue * itemFactorValue - regUser * posUserFactorValue));
                    userFactors.add(negUserIdx, factorIdx, learnRate * (deriValue * (-itemFactorValue) - regUser * negUserFactorValue));

                    loss -= (regUser * itemFactorValue * itemFactorValue + regItem * posUserFactorValue * posUserFactorValue + regItem * negUserFactorValue * negUserFactorValue) * 0.5;
                }
            }
            if (isConverged(iter) && earlyStop) {
                break;
            }
            updateLRate(iter);
        }
    }
    
    private int[] diff(int items, List<Integer> pu) {
		boolean[] resultbool = new boolean[items];
		int[] result = new int[items - pu.size()];
		int j = 0;
		for (int i = 0; i < pu.size(); i++) {
			resultbool[pu.get(i)] = true;
		}
		for (int i = 0; i < resultbool.length; i++) {
			if (!resultbool[i])
				result[j++] = i;
		}
		return result;
	}
    
    private List<Set<Integer>> getItemUsersSet(SparseMatrix sparseMatrix) {
        List<Set<Integer>> itemUsersSet = new ArrayList<>();
        for (int itemIdx = 0; itemIdx < numItems; ++itemIdx) {
        	itemUsersSet.add(new HashSet(sparseMatrix.getRows(itemIdx)));
        }
        return itemUsersSet;
    }
}
