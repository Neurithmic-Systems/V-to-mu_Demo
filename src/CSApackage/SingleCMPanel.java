/*
 * SingleCMPanel.java
 *
 * Created on June 15, 2007, 12:16 PM
 */

package CSApackage;

import static CSApackage.V_to_mu_plot.dashed;
import java.awt.*;
import java.text.NumberFormat;

/**
 *
 * @author  Owner
 */
public class SingleCMPanel extends javax.swing.JPanel
{
  private Mac theMac = null;
  V_to_mu_plot m_Plot_Panel = null;
  
  int mouse_X;
  int mouse_Y;
  private int focusedUnit;
  
  int cellHorizInset = 3;
  int cellDiameter = 14;
  int cellHorizSpace = cellDiameter + 2 * cellHorizInset;
  
  int num_Y_AxisTicks = 5;
  
  float[] V_vals = null;
  float[] muVals = null;
  float[] rhovals = null;
  
  int m_vert_spacer = 25;
  
  int plotHeightPixels = 0;
  int overallHeight = 0;
  
  int plotWidthPixels = 500;
  int plotOrigin_X_Inset = 140;
  int x_AxisRightBufferPixels = 20;
  int overallWidth = plotOrigin_X_Inset + plotWidthPixels + x_AxisRightBufferPixels;
  
  int m_p_plot_height = 0;
  int m_p_plot_bottom = 0;
  int m_p_plot_y_mid = 0;
  
  int m_mu_plot_height = 0;
  int m_mu_plot_bottom = 0;
  int m_mu_plot_y_mid = 0;
  
  int m_V_plot_height = 0;
  int m_V_plot_bottom = 0;  
  int m_V_plot_y_mid = 0;
  
  int m_cell_top = 0;
  
  public NumberFormat m_FloatFormat = NumberFormat.getNumberInstance();
  public NumberFormat m_FloatFormat_prob = NumberFormat.getNumberInstance();
  public NumberFormat m_IntFormat = NumberFormat.getNumberInstance();
  
  MainCSA_demoPanel m_Controller = null;
  private MacPlanPanel m_macPlanPanel = null;
  
  /**
   * Creates new form single CM plan panel
   * 
   */
  public SingleCMPanel()
  {
    m_FloatFormat.setMaximumFractionDigits(1);
    m_FloatFormat.setMinimumFractionDigits(1);
    m_FloatFormat.setMinimumIntegerDigits(1);
    
    m_FloatFormat_prob.setMaximumFractionDigits(2);
    m_FloatFormat_prob.setMinimumFractionDigits(2);
    m_FloatFormat_prob.setMinimumIntegerDigits(1);
    
    m_IntFormat.setMaximumFractionDigits(0);
    
    initComponents();
  }
  
  /** This method is called from within the constructor to
   * initialize the form.
   * WARNING: Do NOT modify this code. The content of this method is
   * always regenerated by the Form Editor.
   */
  // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
  private void initComponents() {

    addMouseMotionListener(new java.awt.event.MouseMotionAdapter() {
      public void mouseMoved(java.awt.event.MouseEvent evt) {
        formMouseMoved(evt);
      }
    });
    setLayout(null);
  }// </editor-fold>//GEN-END:initComponents

  private void formMouseMoved(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_formMouseMoved
    mouse_X = evt.getX();
    mouse_Y = evt.getY();  
    repaint();
    this.m_Controller.get_V_to_mu_Panel().repaint();
  }//GEN-LAST:event_formMouseMoved
  
  public void SetController ( MainCSA_demoPanel controller )
  {
    m_Controller = controller;
  }
  
  public void set_plot_class( V_to_mu_plot plotPanel ) { m_Plot_Panel = plotPanel; }
  
  Font axisValuesFont = new Font("Serif", Font.BOLD, 14);
  Font axisVarFont = new Font("Serif", Font.BOLD, 16);
  Font axisVarFontLarge = new Font("Serif", Font.BOLD | Font.ITALIC, 20);
  Font titleFont = new Font("Serif", Font.BOLD, 20);
          
  protected void paintComponent(java.awt.Graphics g)
  {
    super.paintComponent(g);
    Graphics2D g2 = (Graphics2D) g;
    
    overallHeight = this.getHeight();
    overallWidth = this.getWidth();
    plotWidthPixels = overallWidth - plotOrigin_X_Inset - x_AxisRightBufferPixels;    
    
    if ( m_Plot_Panel == null ) { return; }
    
    m_p_plot_height = (int) ( overallHeight * 0.23 );
    m_mu_plot_height = m_p_plot_height;
    m_V_plot_height = m_p_plot_height;
    
    m_p_plot_bottom = m_p_plot_height + m_vert_spacer; 
    m_mu_plot_bottom = m_p_plot_bottom + m_mu_plot_height + m_vert_spacer; 
    m_V_plot_bottom = m_mu_plot_bottom + m_V_plot_height + m_vert_spacer; 
    
    m_p_plot_y_mid = (int) ( (float) m_p_plot_bottom - m_p_plot_height / 2 );
    m_mu_plot_y_mid = (int) ( (float) m_mu_plot_bottom - m_mu_plot_height / 2 );
    m_V_plot_y_mid = (int) ( (float) m_V_plot_bottom - m_V_plot_height / 2 );
    
    m_cell_top = m_V_plot_bottom + 10;
    
    // Draw background of chart in same color as background of the Focused CM in mac panel chart   
    if (m_Controller != null)
      g2.setColor(m_Controller.focused_CM_background);
    g2.fillRect(plotOrigin_X_Inset, m_vert_spacer, getWidth() - plotOrigin_X_Inset - x_AxisRightBufferPixels, m_p_plot_height);
    g2.fillRect(plotOrigin_X_Inset, m_p_plot_bottom + m_vert_spacer, getWidth() - plotOrigin_X_Inset - x_AxisRightBufferPixels, m_mu_plot_height);
    g2.fillRect(plotOrigin_X_Inset, m_mu_plot_bottom + m_vert_spacer, getWidth() - plotOrigin_X_Inset - x_AxisRightBufferPixels, m_V_plot_height);
    
    g2.setColor( Color.BLACK );
    
    int y = 0;    
    int numCells = theMac.K;
    
    // Compute how wide bars should be to fit everything
    cellHorizSpace = (numCells > 0) ? plotWidthPixels / numCells : cellHorizSpace;
    if (cellHorizSpace > 20)
    {
      cellHorizInset = 3;
      cellDiameter = 14;
      cellHorizSpace = cellDiameter + 2 * cellHorizInset;
    }
    else
    {
      cellHorizInset = (int)(cellHorizSpace/7.0);
      cellDiameter = cellHorizSpace - 2 * cellHorizInset;
    }    
    
    // Draw x-axes for rho, mu, and V plots   
    
    g2.drawLine(plotOrigin_X_Inset, m_p_plot_bottom, getWidth() - x_AxisRightBufferPixels, m_p_plot_bottom );
    g2.setFont(axisVarFontLarge);
    g2.drawString("\u03c1", (int) (plotOrigin_X_Inset / 3) - 3, m_p_plot_y_mid - 32);
    g2.setFont(axisVarFont);
    g2.drawString( "Prob. of", 10, m_p_plot_y_mid);
    g2.drawString( "Winning", 10, m_p_plot_y_mid + 22 );
    g2.drawString( "(normed \u03bc)", 10, m_p_plot_y_mid + 44 );
    
    g2.drawLine(plotOrigin_X_Inset, m_mu_plot_bottom, getWidth() - x_AxisRightBufferPixels, m_mu_plot_bottom );
    g2.setFont(axisVarFontLarge);
    g2.drawString("\u03bc", (int) (plotOrigin_X_Inset / 3) - 3, m_mu_plot_y_mid - 26);
    g2.setFont(axisVarFont);
    g2.drawString( "Rel. Prob.", 10, m_mu_plot_y_mid + 6);
    g2.drawString( "of Winning", 10, m_mu_plot_y_mid + 28 );
    
    g2.drawLine(plotOrigin_X_Inset, m_V_plot_bottom, getWidth() - x_AxisRightBufferPixels, m_V_plot_bottom );
    g2.setFont(axisVarFontLarge);
    g2.drawString("V", (plotOrigin_X_Inset / 3) - 3, m_V_plot_y_mid - 30 );
    g2.setFont(axisVarFont);
    g2.drawString( "Normed", 10, m_V_plot_y_mid - 4);
    g2.drawString( "Synaptic", 10, m_V_plot_y_mid + 18);
    g2.drawString( "Support", 10, m_V_plot_y_mid + 40 );
    
    // draw y-axes for rho, mu, and V plots
    
    g2.drawLine(plotOrigin_X_Inset, m_p_plot_bottom, plotOrigin_X_Inset, m_p_plot_bottom - m_p_plot_height );
    g2.drawLine(plotOrigin_X_Inset, m_mu_plot_bottom, plotOrigin_X_Inset, m_mu_plot_bottom - m_mu_plot_height );
    g2.drawLine(plotOrigin_X_Inset, m_V_plot_bottom, plotOrigin_X_Inset, m_V_plot_bottom - m_V_plot_height );
    
    g2.setFont(axisValuesFont);
    int y_axis_maj_interval = (int) ( m_p_plot_height / (float) num_Y_AxisTicks );
    int y_pos = 0;
    for (int h = 0; h <= num_Y_AxisTicks; h++)                                              // draw y-axis ticks and vals
    {
      y_pos = h * y_axis_maj_interval;
      g2.drawString(m_FloatFormat.format((float) h / num_Y_AxisTicks), plotOrigin_X_Inset - 22, m_p_plot_bottom - y_pos + 6 );
      // the min value of mu is always 1, so special case this.
      if (h == 0)
        g2.drawString("1", plotOrigin_X_Inset - 26, m_mu_plot_bottom - y_pos + 6 );
      else
        g2.drawString(m_IntFormat.format((float) h / num_Y_AxisTicks * theMac.max_V_to_mu_Multiplier), plotOrigin_X_Inset - 26, m_mu_plot_bottom - y_pos + 6 );
      g2.drawString(m_FloatFormat.format((float) h / num_Y_AxisTicks), plotOrigin_X_Inset - 22, m_V_plot_bottom - y_pos + 6 );      
    }
    
    g2.drawString("Cell", 30, m_cell_top + 15);
    
    //// draw cells and bars
    
    int focusedCM = m_macPlanPanel.getFocused_CM_Index();
    int winnerIndex = theMac.getWinningIndex(focusedCM);
    int max_V_Index = theMac.getMax_V_Index(focusedCM);        
    double divisor = ( theMac.muSum.get(focusedCM) > 0 ) ?  theMac.muSum.get(focusedCM) : 1;
      
    /// draw cells and mu and rho bars.  And show hovering values if cursor is over in
    /// a vertical region for a paricular cell.
    
    float barHeightCorrection;
    int x_left = 0;  // convenience var in loop    
    for (int c = 0; c < numCells; c++)
    {      
      x_left = plotOrigin_X_Inset + c * cellHorizSpace + cellHorizInset;
      
      // Set color for cell and mu and rho bar
      if (c == winnerIndex && c == max_V_Index)
        g2.setColor(m_Controller.correctWinColor);
      else if (c == winnerIndex && c != max_V_Index)
        g2.setColor(m_Controller.incorrectWinColor);
      else
        g2.setColor(m_Controller.irrelevantColor);         
      
      g2.fillOval(x_left, m_cell_top, cellDiameter, cellDiameter ); 
      
      y = (int) ( theMac.get_specific_mu_val(focusedCM, c) / divisor * m_p_plot_height );
      g2.fillRect(x_left, m_p_plot_bottom - y, cellDiameter, y );      
      
      barHeightCorrection = theMac.getEta() / theMac.max_V_to_mu_Multiplier;
      y = (int) (theMac.get_specific_mu_val(focusedCM, c) / theMac.getEta() * barHeightCorrection * m_mu_plot_height);
      g2.fillRect(x_left, m_mu_plot_bottom - y, cellDiameter, y );
      
      // Show values as mouse hovers over bars
      if (mouse_X >= x_left && mouse_X <= x_left + cellDiameter) // && mouse_Y >= y_pos-3 && mouse_Y <= y_pos+3 )
      {
        setFocusedUnit(c);
        g2.setColor( Color.black );
        g2.drawString(m_FloatFormat_prob.format(theMac.get_specific_mu_val(focusedCM, c) / theMac.muSum.get(focusedCM)), x_left - 8, m_p_plot_bottom - m_p_plot_height - 8 ); 
        g2.drawString(m_FloatFormat_prob.format(theMac.get_specific_mu_val(focusedCM, c)), x_left - 8, m_mu_plot_bottom - m_mu_plot_height - 8 ); 
        g2.drawString(m_FloatFormat_prob.format(theMac.get_specific_V_val(focusedCM, c)), x_left, m_V_plot_bottom - m_V_plot_height - 8 );         
      }      
    }
    
    /// Draw V bars
    
    for (int c = 0; c < numCells; c++)
    {      
      x_left = plotOrigin_X_Inset + c * cellHorizSpace + cellHorizInset;      
      // Set color for V bar
      g2.setColor((c == max_V_Index) ? m_Controller.correctWinColor : m_Controller.irrelevantColor);            
      y = (int) ( theMac.get_specific_V_val(focusedCM, c) * m_V_plot_height );
      g2.fillRect(x_left, m_V_plot_bottom - y, cellDiameter, y );
    }
    
    //// Draw faint horizontal lines on the V plot to show crosstalk limits.
    
    int y_max_V_in_pixels = 0;
    if (m_Controller != null)
    {
      Stroke oldStroke = g2.getStroke();
      g2.setStroke(dashed);

      if (m_Controller.isCrossTalkRelativeToCurrentMax_V())
        y_max_V_in_pixels = (int) (m_V_plot_height * theMac.GetWinner_V_Val());
      else
        y_max_V_in_pixels = (int) (m_V_plot_height * 1);
      
      g2.setColor( m_Controller.colorLowCrosstalkLimit );
      int min_crosstalk_y = m_V_plot_bottom - (int)(theMac.GetCrossTalkLowLimFactor() * y_max_V_in_pixels);
      g2.drawLine(plotOrigin_X_Inset, min_crosstalk_y, getWidth() - x_AxisRightBufferPixels, min_crosstalk_y );
      
      g2.setColor( m_Controller.colorHighCrosstalkLimit );
      int max_crosstalk_y = m_V_plot_bottom - (int)(theMac.GetCrossTalkHighLimFactor() * y_max_V_in_pixels);
      g2.drawLine(plotOrigin_X_Inset, max_crosstalk_y, getWidth() - x_AxisRightBufferPixels, max_crosstalk_y ); 
      g2.setStroke(oldStroke);
    }
    
  }

  /**
   * @return the theMac
   */
  public Mac getTheMac() {
    return theMac;
  }

  /**
   * @param theMac the theMac to set
   */
  public void setTheMac(Mac theMac) {
    this.theMac = theMac;
  }

  /**
   * @return the m_macPlanPanel
   */
  public MacPlanPanel getM_macPlanPanel() {
    return m_macPlanPanel;
  }

  /**
   * @param m_macPlanPanel the m_macPlanPanel to set
   */
  public void setM_macPlanPanel(MacPlanPanel m_macPlanPanel) {
    this.m_macPlanPanel = m_macPlanPanel;
  }

    /**
     * @return the focusedUnit
     */
    public int getFocusedUnit() {
        return focusedUnit;
    }

    /**
     * @param focusedUnit the focusedUnit to set
     */
    public void setFocusedUnit(int focusedUnit) {
        this.focusedUnit = focusedUnit;
    }
  
  // Variables declaration - do not modify//GEN-BEGIN:variables
  // End of variables declaration//GEN-END:variables
  
}
