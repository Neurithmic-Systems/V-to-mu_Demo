/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package CSApackage;

import java.util.ArrayList;
import java.util.List;

/**
 * This is the class that holds the actual model of the mac and the fns for creating the V's, mu's, etc.
 * 
 * @author rod
 */
public class Mac {  
  
  private CSAdemo theApp = null;
  
  static final int default_Q = 8;
  protected int Q = default_Q;
  static final int default_K = 4;
  protected int K = default_K;
  final int max_K = 20;
  
  float G = 1.0f;
  private int horizInflectionLocation = 50;
  private float eta = 0;                                        // height of V-to-mu sigmoid transform
  private float eccentricity = 15;                              // eccentricity of V-to-mu sigmoid transform   Also Called "beta" in the code
  private float gamma = 2.0f;                                   // exponent in eta equation..determines how quickly eta rises as fn of G.
  static final int max_V_to_mu_Multiplier = 400;
  private int V_to_mu_Multiplier = max_V_to_mu_Multiplier;
  
  // These arrays hold the (x,y), i.e., (V,mu), values specifying the whole sigmoid fn.
  private int num_whole_sigmoid_points = 200;  
  private float [] whole_sigmoid_V = new float[ num_whole_sigmoid_points ];
  private float [] whole_sigmoid_mu = new float[ num_whole_sigmoid_points ];
  
  // These lists hold the values pertinent to the actual cells of the mac.
  List<List<Double>> V = null;
  List<List<Double>> mu = null;  
  List<List<Double>> rho = null;
  List<List<Double>> rhoCumulative = null; 
  List<Double> muSum = null;
  List<Integer> maxV_index = null;  
  List<Double> second_highest_V_val = null;
  List<Integer> winnerIndex = null; 
  
  private float muMax = 0;
  private float muMin = 0;
  
  float G_lowCutoff = 0.0f;
  float G_highCutoff = 1.0f;
  
  protected double accuracy = 0;
  double expectedAccuracy = 0;
  double varianceAccuracy = 0;
  double stdDevExpectedAccuracy = 0;
  
  double crosstalk_V_upper_lim_factor = 0.1f;
  double crosstalk_V_lower_lim_factor = 0;
  
  double crosstalk_V_upper_lim = 0.1f;
  double crosstalk_V_lower_lim = 0;
  
  double winner_V_val = 1.0f;
  double max_V_cell_WinProb = 0;
  
  
  public Mac()
  {
    for ( int x = 0; x < whole_sigmoid_V.length; x++ )                                      // Fill out whole_sigmoid_V vals array
    {
      whole_sigmoid_V[ x ] = (float) x / (float) num_whole_sigmoid_points;
    }
    
    V = new ArrayList<>(Q);
    mu = new ArrayList<>(Q);
    rho = new ArrayList<>(Q);
    rhoCumulative = new ArrayList<>(Q);
        
    for (int q = 0; q < Q; q++)
    {
      List<Double> cellsOfCM = new ArrayList<>(K);
      V.add(cellsOfCM);
      cellsOfCM = new ArrayList<>(K);
      mu.add(cellsOfCM);
      cellsOfCM = new ArrayList<>(K);
      rho.add(cellsOfCM);
      cellsOfCM = new ArrayList<>(K);
      rhoCumulative.add(cellsOfCM);
    }
    
    this.ensureArraySizes(K);
    
    maxV_index = new ArrayList<>(Q);
    muSum = new ArrayList<>(Q);
    second_highest_V_val = new ArrayList<>(Q);
    winnerIndex = new ArrayList<>(Q);
    for (int cm = 0; cm < Q; cm++)
    {
      maxV_index.add(0);
      muSum.add(0d);
      second_highest_V_val.add(0d);
      winnerIndex.add(0);
    }    
    
    calculate_eta();
    
    SetCrossTalkLowLimFactor( 0 );
    SetCrossTalkHighLimFactor( 0.5f );
    SetWinner_V_Val( 1.0f );
  }
  
  /**
   * The param, numCells, comes from V-to-mu plot panel class where it is updated based on user clicks.
   * 
   * @param numCells
   * @return 
   */
  protected void ensureArraySizes(int numCells)
  {           
    int numCMsToAdd = (V.size() < Q) ? Q - V.size() : 0;    
    
    // First see if the outer dim is correct. The user may have changed Q. We assume the dims
    // of the mu and V array are always the same.
    
    if (numCMsToAdd > 0)
    {
      for (int q = 0; q < numCMsToAdd; q++)
      {
        List<Double> cellsOfCM = new ArrayList<>(numCells);
        V.add(cellsOfCM);
        cellsOfCM = new ArrayList<>(numCells);
        mu.add(cellsOfCM);
        cellsOfCM = new ArrayList<>(numCells);
        rho.add(cellsOfCM);
        cellsOfCM = new ArrayList<>(numCells);
        rhoCumulative.add(cellsOfCM);
        maxV_index.add(0);
        muSum.add(0d);
        second_highest_V_val.add(0d);
        winnerIndex.add(0);
      }      
    }
    
    for (int q = 0; q < Q; q++)
    {
      List<Double> innerList_V = V.get(q);
      List<Double> innerList_mu = mu.get(q);
      List<Double> innerList_rho = rho.get(q);
      List<Double> innerList_rhoCum = rhoCumulative.get(q);
      while (innerList_V.size() < numCells)
      {
        innerList_V.add(0d);
        innerList_mu.add(0d);
        innerList_rho.add(0d);
        innerList_rhoCum.add(0d);
      }    
    }       
    
    K = numCells;       
  }
  
  /**
   * Called when user clicks in V-to-mu plot.  It sets the V value for the newly
   * added cell in all CMs to be the V determined by where the user clicked in the plot.  
   */
  public void specifyNewCell_Vs(int new_cell_index, float specified_V)
  {
    V.get(0).set(new_cell_index, (double) specified_V);
    
    for (int q = 0; q < Q; q++)   // loop over CMs
    {
      V.get(q).set(new_cell_index, (double) specified_V);
    }
  }
  
  /**
   * Not sure why I originally decided to add new V values in this particular way, but
   * the new version above seems to make more sense, at least from a pedagogical standpoint.
   * 
   * @param new_cell_index
   * @param specified_V 
   */
  public void specifyNewCell_Vs_OLD(int new_cell_index, float specified_V)
  {
    V.get(0).set(new_cell_index, (double) specified_V);
    
    double crosstalkRange = crosstalk_V_upper_lim - crosstalk_V_lower_lim;
    double val = 0;
    
    for (int q = 1; q < Q; q++)   // loop over CMs
    {
      val = Math.random() * crosstalkRange + crosstalk_V_lower_lim;
      V.get(q).set(new_cell_index, (double) val);
    }
  }
  
  /** 
   * Determines parameter, eta, a multiplier that affects the range of the V-to-mu function. 
   * G is always between 0 and 1.  We raise G to gamma.  The higher is gamma, the the lower is eta.
   */
  public float calculate_eta()
  {
    float temp1 = 0;
    float temp2 = 1;
    
    if ( G < G_lowCutoff )
    {
      eta = 0;
    }
    else
    {
      temp1 = G - G_lowCutoff;
      temp2 = 1.0f - G_lowCutoff;
      
      // the highest "temp1 / temp2" can be is 1.  gamma is an int >= 1.  So result of raising to power gamma
      // is that the higher gamma is, the more convex the map from V to mu is. So higher gamma
      // results in a more stringent matching fn (a tighter generalization gradient).
      // V_to_mu_Multiplier just magnifies the range of possible mu values.
      eta = (float) Math.pow( ( temp1 / temp2 ), gamma ) * V_to_mu_Multiplier;
    }
    
    return eta;
  }
  
  /** 
   * Creates random distributions of V values in each of the CMs.  One cell will be chosen at
   * random in each CM to have a max V, V_max, controlled by a slider.  The other cells in each CM will be 
   * assigned random V values uniformly distributed in a range, whose parameters vary to simulate conditions 
   * that would exist at various periods of the model's life.
   */
  public void create_V_Distributions(boolean pick_new_max_V_dex, boolean randomly_draw_non_max_V_cells)
  {        
    int declaredMax_V_index = 0; 
    double val = 0;
    
    if (theApp.getMain_CSA_panel() != null)
    {
      if ( theApp.getMain_CSA_panel().isCrossTalkRelativeToCurrentMax_V())
      {
        crosstalk_V_lower_lim = crosstalk_V_lower_lim_factor * winner_V_val;
        crosstalk_V_upper_lim = crosstalk_V_upper_lim_factor * winner_V_val;
      }
      else
      {
        crosstalk_V_lower_lim = crosstalk_V_lower_lim_factor * 1;
        crosstalk_V_upper_lim = crosstalk_V_upper_lim_factor * 1;
      }
    }
    double crosstalkRange = crosstalk_V_upper_lim - crosstalk_V_lower_lim;
    
    for (int q = 0; q < Q; q++)   // loop over CMs
    {
      if (pick_new_max_V_dex)
      {
        declaredMax_V_index = (int) ( Math.random() * K );                      // randomly choose one cell to have the max V value in the CM, winner_V_val.   
        maxV_index.set(q, declaredMax_V_index);
      }
      // otherwise we don't update max_V_Index[q].  But of course, we do update the V value of that cell.
      
      for (int c = 0; c < K; c++)    // loop over K cells of a CM
      {
        if (c == maxV_index.get(q))
        {
          val = winner_V_val;                                                   // winner_V_val is pre-defined, but can be controlled by sliders
          V.get(q).set(c, val);
        }
        else if (randomly_draw_non_max_V_cells)   // else pick a random V value in the current crosstalk range
        {
          val = Math.random() * crosstalkRange + crosstalk_V_lower_lim;
          V.get(q).set(c, val);
        }
        // use current V value for cell        
      }   
    }  
  }
  
  /** Calculates the V-to-mu curve.  This uses a different, more standard, sigmoid model, i.e. 
   * 
   * S(x) = eta * 1 / (1 + exp(-(x-inflection_pt) / eccentricity))
   * 
   * I added this version because the earlier version, which used the 'correction' term was never quite right.
   * The sigmoid would wiggle in a funny way as you slid it left to right.
   * 
   * Note that in this version, we don't really need to do transform in two stages, i.e., through mu.  We can
   * deal directly from V to rho.
   * 
   * So far, in Sparsey, I've always just manipulated sigmoid ht, eta, as function of G.  But I really think
   * that it would be better to simultaneously vary eta and eccentricity as fn of G.  
   * Varying eta as fn of G has always had a straightforward neural interpretation, i.e., simply corresponding
   * to varying the amount of noise (relative to signal) present in the winner choice process.  I need to think
   * through the neural implications of adding simultaneous modulation of eccentricity.
   */  
  public void compute_whole_sigmoid()
  {
    for (int x = 0; x < num_whole_sigmoid_points; x++)
    {
      whole_sigmoid_mu[x] = 100 * whole_sigmoid_V[x];
      whole_sigmoid_mu[x] = (float) Math.exp(-1 * (whole_sigmoid_mu[x] - horizInflectionLocation) / eccentricity) + 1;
      whole_sigmoid_mu[x] = eta / whole_sigmoid_mu[x] + 1;

      if (whole_sigmoid_mu[x] < muMin)
        muMin = whole_sigmoid_mu[x];

      if (whole_sigmoid_mu[x] > muMax)
        muMax = whole_sigmoid_mu[x];
    }
  }  
  
  /**
   * Called when sliders are moved.  Updates the mu vals, muSums, cumulative rho vals, and rho distributions
   * based on the V distribution.
   */
  public void updateDependentDistributions()
  {    
    double temp = 0;       
    for (int q = 0; q < Q; q++)
    {
      muSum.set(q, 0d);             
      for (int c = 0; c < K; c++)
      {        
        // Compute mu vals (relative likelihoods) and muSum. rho values are computed in next loop.
        // We also fill out the cumulative rho distribution in this loop as well. We need it so 
        // we can make a draw of a winner in each CM.
        
        temp = 100 * V.get(q).get(c);      
        temp = (float) Math.exp(-1 * (temp - horizInflectionLocation) / eccentricity) + 1;        // "1" needed to prevent div by 0 in next line.
        temp = eta / temp  + 1;
        
        mu.get(q).set(c, temp);
        muSum.set(q, muSum.get(q) + temp);
        rhoCumulative.get(q).set(c, muSum.get(q));
      }       
     
      // Determine rho vals by normalizing mu vals
      for (int c = 0; c < K; c++)    // loop over K cells of a CM
        rho.get(q).set(c, mu.get(q).get(c) / muSum.get(q));
    }  
  }
  
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  //// OLD WAY FOR COMPUTING SIGMOID, THAT INVOLVED A CORRECTION FACTOR....
  
//   m_correction = 6 * Math.abs(horizInflectionLocation - 0.5f ) + 1;
//    
//    // compute whole_sigmoid_mu vals of every pt on curve, so we can draw the curve.
//    for (int x = Highest_X_IndexWith_V_EqualsZero; x < Lowest_X_IndexWith_V_EqualsMax; x++)
//    {
//      whole_sigmoid_mu[x] = whole_sigmoid_V[x] - horizInflectionLocation;
//      whole_sigmoid_mu[x] = (float) Math.exp(-eccentricity * m_correction * whole_sigmoid_mu[x]) + 1;
//      whole_sigmoid_mu[x] = eta / whole_sigmoid_mu[x];
//    }
  
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  public void chooseCodeAndComputeAccuracies()
  {
    // We'll use these accumulators then just divide by Q to get frac. acc. vals. 
    accuracy = 0;     
    expectedAccuracy = 0;
    varianceAccuracy = 0;
    
    for (int q = 0; q < Q; q++)
    {
      winnerIndex.set(q, pickWinner(q));          // pick winner by draw from cumulative rho dist.
      
      if (maxV_index.get(q) == winnerIndex.get(q))
        accuracy += 1d;
      // else it's a wrong pick so don't add anything to accuracy accumulator
      
      max_V_cell_WinProb = rho.get(q).get( maxV_index.get(q) );
      expectedAccuracy += max_V_cell_WinProb;    
      varianceAccuracy += max_V_cell_WinProb * (1 - max_V_cell_WinProb); 
    }
    accuracy /= (double) Q;
    expectedAccuracy /= (double) Q;
    varianceAccuracy /= (double) Q;
    stdDevExpectedAccuracy = Math.sqrt(varianceAccuracy);
  }
  
  /** 
   * All CMs have same stats. Compute prob that the V=1 cell wins in a CM.  The K-1 other
   * cells in a CM have probs chosen uniformly from a specified range.  So the prob that
   * V=1 cell wins, in a given CM, is 1 / (sum of all K unormalized prob vals).  Call that P.  
   * Since we have Q independent CMs, we have a binomial dist., with mean = Q * P and
   * variance = Q * P * (1 - P).
   */
  public void computeCodeAccuracyStats()
  {    
    // the prob that the max V cell wins in each CM = mu val of that cell over muSum.
    // Call that value, P.  the overall accuracy across all CMs, as a percentage, is also P, because all CMs have
    // approx the same value for P.  The expected number of CMs in which the max V cell wins is just P * Q, but we don't explicitly report that.
    
//    this.expectedAccuracy = this.expectedAccuracy;                                // mean of binomial dist. is Q times this.
    varianceAccuracy = expectedAccuracy * (1 - expectedAccuracy);      // var of binomial dis. is Q times this.
    stdDevExpectedAccuracy = Math.sqrt(this.varianceAccuracy);
  }
  
  /**
   * pick winner in CM q as a softmax.  It's actually done using the cum rho dist.
   * @param q
   * @return 
   */
  public int pickWinner(int q)
  {
    double RandVal = Math.random() * muSum.get(q);
    int dex = 0;
    while (RandVal > rhoCumulative.get(q).get(dex))    
      dex++;          
    return dex;
  }
  
  /**
   * @return the Q
   */
  public int getQ() { return Q; }  
  public int getK() { return K; }

  /**
   * Whenever we set Q, we should also make sure that the num of cols 
   * is big enough to show the charts for all the CMs.
   * 
   * @param Q the Q to set
   */
  public void setQ(int Q)
  {
    this.Q = Q;
    ensureArraySizes(K);    
  }
  
  /**
   * Whenever we set K, we should also.....
   * 
   * @param K the K to set
   */
  public void setK(int K)
  {
    this.K = K;
    ensureArraySizes(K);    
  }
  
  /**
   * We're defining accuracy here as the fraction of CMs in which the winning cell,
   * which is chosen from the rho distribution, is the same as the max-V cell.  This
   * is based on a single draw of a code.
   * 
   * @return 
   */
  public double getAccuracy() { return accuracy; }  
  public double getExpectedAccuracy() { return this.expectedAccuracy; }  
  public double getVarianceExpectedAccuracy() { return this.varianceAccuracy; }  
  public double getStdDevExpectedAccuracy() { return this.stdDevExpectedAccuracy; }
  
  public void reset_V_maxes()
  {
    for ( int cm = 0; cm < Q; cm++ )
    {
      maxV_index.set(cm, 0);
    }
  }
  
  public void ClearCells()
  {
    K = 0;
  }
  
  public float get_specific_V_val (int q, int k) { return V.get(q).get(k).floatValue(); }  
  public void set_specific_V_val (int q, int k, double val) { V.get(q).set(k, val); }
  
  public float get_specific_mu_val (int q, int k) { return mu.get(q).get(k).floatValue(); }  
  public void set_specific_mu_val (int q, int k, double val) { mu.get(q).set(k, val); }
  
  public int getWinningIndex(int cm) { return winnerIndex.get(cm); }  
  public int getMax_V_Index(int cm) { return maxV_index.get(cm); }
  
  public void SetCrossTalkHighLimFactor( float val ) { crosstalk_V_upper_lim_factor = val; }  
  public float GetCrossTalkHighLimFactor() { return (float)crosstalk_V_upper_lim_factor; }  
  public void SetCrossTalkHighLim( float val ) { crosstalk_V_upper_lim = val; }
  public float GetCrossTalkHighLim() { return (float)crosstalk_V_upper_lim; }
  
  public void SetCrossTalkLowLimFactor( float val ) { crosstalk_V_lower_lim_factor = val; }  
  public float GetCrossTalkLowLimFactor() { return (float)crosstalk_V_lower_lim_factor; }
  public void SetCrossTalkLowLim( float val ) { crosstalk_V_lower_lim = val; }
  public float GetCrossTalkLowLim() { return (float)crosstalk_V_lower_lim; }
  
  public void SetWinner_V_Val( float val ) { winner_V_val = val; }  
  public float GetWinner_V_Val() { return (float)winner_V_val; }
  
  public CSAdemo getTheApp() { return theApp; }
  public void setTheApp(CSAdemo theApp) { this.theApp = theApp; }

  /**
   * @return the muMax
   */
  public float getMuMax() {
    return muMax;
  }

  /**
   * @param muMax the muMax to set
   */
  public void setMuMax(float muMax) {
    this.muMax = muMax;
  }

  /**
   * @return the muMin
   */
  public float getMuMin() {
    return muMin;
  }

  /**
   * @param muMin the muMin to set
   */
  public void setMuMin(float muMin) {
    this.muMin = muMin;
  }

  /**
   * @return the whole_sigmoid_V
   */
  public float[] getWhole_sigmoid_V() {
    return whole_sigmoid_V;
  }

  /**
   * @param whole_sigmoid_V the whole_sigmoid_V to set
   */
  public void setWhole_sigmoid_V(float[] whole_sigmoid_V) {
    this.whole_sigmoid_V = whole_sigmoid_V;
  }

  /**
   * @return the whole_sigmoid_mu
   */
  public float[] getWhole_sigmoid_mu() {
    return whole_sigmoid_mu;
  }

  /**
   * @param whole_sigmoid_mu the whole_sigmoid_mu to set
   */
  public void setWhole_sigmoid_mu(float[] whole_sigmoid_mu) {
    this.whole_sigmoid_mu = whole_sigmoid_mu;
  }

  /**
   * @return the num_whole_sigmoid_points
   */
  public int getNum_whole_sigmoid_points() {
    return num_whole_sigmoid_points;
  }

  /**
   * @param num_whole_sigmoid_points the num_whole_sigmoid_points to set
   */
  public void setNum_whole_sigmoid_points(int num_whole_sigmoid_points) {
    this.num_whole_sigmoid_points = num_whole_sigmoid_points;
  }

  /**
   * @return the eta
   */
  public float getEta() {
    return eta;
  }

  /**
   * @param eta the eta to set
   */
  public void setEta(float eta) {
    this.eta = eta;
  }

  /**
   * @return the eccentricity
   */
  public float getEccentricity() {
    return eccentricity;
  }

  /**
   * @param eccentricity the eccentricity to set
   */
  public void setEccentricity(float eccentricity) {
    this.eccentricity = eccentricity;
  }

  /**
   * @return the horizInflectionLocation
   */
  public int getHorizInflectionLocation() {
    return horizInflectionLocation;
  }

  /**
   * @param horizInflectionLocation the horizInflectionLocation to set
   */
  public void setHorizInflectionLocation(int horizInflectionLocation) {
    this.horizInflectionLocation = horizInflectionLocation;
  }

  /**
   * @return the gamma
   */
  public float getGamma() {
    return gamma;
  }

  /**
   * @param gamma the gamma to set
   */
  public void setGamma(float gamma) {
    this.gamma = gamma;
  }

  /**
   * @return the V_to_mu_Multiplier
   */
  public int getV_to_mu_Multiplier() {
    return V_to_mu_Multiplier;
  }

  /**
   * @param V_to_mu_Multiplier the V_to_mu_Multiplier to set
   */
  public void setV_to_mu_Multiplier(int V_to_mu_Multiplier) {
    this.V_to_mu_Multiplier = V_to_mu_Multiplier;
  }
  
}
