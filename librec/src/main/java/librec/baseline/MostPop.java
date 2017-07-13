package librec.baseline;

import java.util.HashMap;
import java.util.Map;
import librec.data.SparseMatrix;
import librec.intf.Recommender;

public class MostPop extends Recommender
{
  private Map<Integer, Integer> userPops;
  private Map<Integer, Integer> itemPops;

  public MostPop(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold)
  {
    super(trainMatrix, testMatrix, fold);

    setAlgoName("MostPop");
  }

  protected void initModel()
  {
    if (isRankingPred)
      this.itemPops = new HashMap();
    else
      this.userPops = new HashMap();
  }

  protected double ranking(int u, int j)
  {
    if (isRankingPred) {
      if (!this.itemPops.containsKey(Integer.valueOf(j))) {
        this.itemPops.put(Integer.valueOf(j), Integer.valueOf(this.trainMatrix.columnSize(j)));
      }
      return ((Integer)this.itemPops.get(Integer.valueOf(j))).intValue();
    }
    if (!this.userPops.containsKey(Integer.valueOf(u))) {
      this.userPops.put(Integer.valueOf(u), Integer.valueOf(this.trainMatrix.rowSize(u)));
    }
    return ((Integer)this.userPops.get(Integer.valueOf(u))).intValue();
  }
}