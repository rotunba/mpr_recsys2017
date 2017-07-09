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
package net.librec.main;

import net.librec.conf.Configuration;
import net.librec.conf.Configuration.Resource;
import net.librec.data.DataModel;
import net.librec.data.model.TextDataModel;
import net.librec.eval.RecommenderEvaluator;
import net.librec.eval.rating.MAEEvaluator;
import net.librec.filter.GenericRecommendedFilter;
import net.librec.filter.RecommendedFilter;
import net.librec.job.RecommenderJob;
import net.librec.recommender.Recommender;
import net.librec.recommender.RecommenderContext;
import net.librec.recommender.cf.UserKNNRecommender;
import net.librec.similarity.PCCSimilarity;
import net.librec.similarity.RecommenderSimilarity;
import java.util.List;

/**
 */
public class Driver {

	public static void main(String[] args) throws Exception {
		
		// recommender configuration
		Configuration conf = new Configuration();
		Resource resource = new Resource("rec/cf/userknn-test.properties");
		conf.addResource(resource);

		// build data model
		DataModel dataModel = new TextDataModel(conf);
		dataModel.buildDataModel();
		
		// set recommendation context
		RecommenderContext context = new RecommenderContext(conf, dataModel);
		RecommenderSimilarity similarity = new PCCSimilarity();
		similarity.buildSimilarityMatrix(dataModel);
		context.setSimilarity(similarity);

		// training
		Recommender recommender = new UserKNNRecommender();
		recommender.recommend(context);

		// evaluation
		RecommenderEvaluator evaluator = new MAEEvaluator();
		recommender.evaluate(evaluator);

		// recommendation results
		List recommendedItemList = recommender.getRecommendedList();
		RecommendedFilter filter = new GenericRecommendedFilter();
		recommendedItemList = filter.filter(recommendedItemList);
		
		//RecommenderJob rj = new RecommenderJob(conf);
		//rj.runJob();
	}

}
