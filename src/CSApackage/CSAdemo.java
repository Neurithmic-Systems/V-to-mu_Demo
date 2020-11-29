/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package CSApackage;

import java.awt.Color;
import java.awt.Point;
import javax.swing.JColorChooser;

/**
 *
 * @author rod
 */
public class CSAdemo extends javax.swing.JFrame {
  
  protected Mac theMac = null;
  
  /**
   * Creates new form CSAdemo
   */
  public CSAdemo() {
    theMac = new Mac();
    theMac.setTheApp(this);
    initComponents();
  }

  /**
   * This method is called from within the constructor to initialize the form.
   * WARNING: Do NOT modify this code. The content of this method is always
   * regenerated by the Form Editor.
   */
  @SuppressWarnings("unchecked")
  // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
  private void initComponents() {

    jSeparator2 = new javax.swing.JSeparator();
    jToolBar1 = new javax.swing.JToolBar();
    instructionsBtn = new javax.swing.JButton();
    jSeparator3 = new javax.swing.JToolBar.Separator();
    runExperimentBtn = new javax.swing.JButton();
    jSeparator1 = new javax.swing.JToolBar.Separator();
    showHoveringValsChkBx = new javax.swing.JCheckBox();
    jSeparator4 = new javax.swing.JToolBar.Separator();
    jLabel1 = new javax.swing.JLabel();
    V_to_mu_fn_thickness_spinner = new javax.swing.JSpinner();
    V_to_mu_fn_thickness_spinner.setValue(3);
    jSeparator5 = new javax.swing.JToolBar.Separator();
    jButton1 = new javax.swing.JButton();
    tie_G_eccent_sliders_Chkbx = new javax.swing.JCheckBox();
    mainCSA_demoPanel1 = new MainCSA_demoPanel(this);
    jMenuBar1 = new javax.swing.JMenuBar();
    helpMenu = new javax.swing.JMenu();

    setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);
    setTitle("Sparsey Code Selection Algorithm (CSA) Demo");
    setMinimumSize(new java.awt.Dimension(700, 600));
    setPreferredSize(new java.awt.Dimension(1200, 1100));

    jToolBar1.setBorder(javax.swing.BorderFactory.createLineBorder(new java.awt.Color(0, 0, 0)));
    jToolBar1.setRollover(true);
    jToolBar1.setPreferredSize(new java.awt.Dimension(13, 30));

    instructionsBtn.setFont(new java.awt.Font("Tahoma", 0, 14)); // NOI18N
    instructionsBtn.setText("Instructions");
    instructionsBtn.setFocusable(false);
    instructionsBtn.setHorizontalTextPosition(javax.swing.SwingConstants.CENTER);
    instructionsBtn.setVerticalTextPosition(javax.swing.SwingConstants.BOTTOM);
    instructionsBtn.addActionListener(new java.awt.event.ActionListener() {
      public void actionPerformed(java.awt.event.ActionEvent evt) {
        instructionsBtnActionPerformed(evt);
      }
    });
    jToolBar1.add(instructionsBtn);
    jToolBar1.add(jSeparator3);

    runExperimentBtn.setBackground(new java.awt.Color(204, 204, 255));
    runExperimentBtn.setFont(new java.awt.Font("Tahoma", 0, 14)); // NOI18N
    runExperimentBtn.setText("Run Experiment");
    runExperimentBtn.setFocusable(false);
    runExperimentBtn.setHorizontalTextPosition(javax.swing.SwingConstants.CENTER);
    runExperimentBtn.setVerticalTextPosition(javax.swing.SwingConstants.BOTTOM);
    runExperimentBtn.addActionListener(new java.awt.event.ActionListener() {
      public void actionPerformed(java.awt.event.ActionEvent evt) {
        runExperimentBtnActionPerformed(evt);
      }
    });
    jToolBar1.add(runExperimentBtn);
    jToolBar1.add(jSeparator1);

    showHoveringValsChkBx.setFont(new java.awt.Font("Tahoma", 0, 14)); // NOI18N
    showHoveringValsChkBx.setSelected(true);
    showHoveringValsChkBx.setText("Show Hovering Vals");
    showHoveringValsChkBx.setToolTipText("");
    showHoveringValsChkBx.setFocusable(false);
    showHoveringValsChkBx.setHorizontalTextPosition(javax.swing.SwingConstants.RIGHT);
    showHoveringValsChkBx.addActionListener(new java.awt.event.ActionListener() {
      public void actionPerformed(java.awt.event.ActionEvent evt) {
        showHoveringValsChkBxActionPerformed(evt);
      }
    });
    jToolBar1.add(showHoveringValsChkBx);
    jToolBar1.add(jSeparator4);

    jLabel1.setFont(new java.awt.Font("Tahoma", 0, 14)); // NOI18N
    jLabel1.setText("  V-to-mu fn thickness  ");
    jToolBar1.add(jLabel1);

    V_to_mu_fn_thickness_spinner.setFont(new java.awt.Font("Tahoma", 0, 14)); // NOI18N
    V_to_mu_fn_thickness_spinner.setToolTipText("set thickness of V-to-my fn graph");
    V_to_mu_fn_thickness_spinner.setMaximumSize(new java.awt.Dimension(100, 30));
    V_to_mu_fn_thickness_spinner.addChangeListener(new javax.swing.event.ChangeListener() {
      public void stateChanged(javax.swing.event.ChangeEvent evt) {
        V_to_mu_fn_thickness_spinnerStateChanged(evt);
      }
    });
    jToolBar1.add(V_to_mu_fn_thickness_spinner);

    jSeparator5.setSeparatorSize(new java.awt.Dimension(10, 0));
    jToolBar1.add(jSeparator5);

    jButton1.setBackground(new java.awt.Color(204, 204, 255));
    jButton1.setFont(new java.awt.Font("Tahoma", 0, 14)); // NOI18N
    jButton1.setText("V-to-mu fn color");
    jButton1.setFocusable(false);
    jButton1.setHorizontalTextPosition(javax.swing.SwingConstants.CENTER);
    jButton1.setVerticalTextPosition(javax.swing.SwingConstants.BOTTOM);
    jButton1.addActionListener(new java.awt.event.ActionListener() {
      public void actionPerformed(java.awt.event.ActionEvent evt) {
        jButton1ActionPerformed(evt);
      }
    });
    jToolBar1.add(jButton1);

    tie_G_eccent_sliders_Chkbx.setFont(new java.awt.Font("Tahoma", 0, 14)); // NOI18N
    tie_G_eccent_sliders_Chkbx.setText("Tie G, V, and Eccen");
    tie_G_eccent_sliders_Chkbx.setToolTipText("<html>\nWhen tied, as one moves G (or V) slider, the eccentricity slider will also move, in a coordinated way, so that the overall relation <br>\nbetween G (input familiarity) and expected code intersection is close to linear. <br><br>\n\nG directly controls the range of the V-to-&mu; transform.  If the range is the only thing we change, then the overall relation <br>\nfrom G to expected code intersection, while correlated, cannot be linear over its whole range.  We need to simultaneously change <br>\nboth the transforms's range and its eccentricity in order to achieve near linearity over all or most of G's range, i.e., [0,1].<br><br>\n\nThere could be other combinations of sigmoid params that can also be changed in coordinated fashion to achieve an overall linear relation <br>\nfrom G to expected code intersection, e.g., involving the horizontal position of the inflection pt, but I haven't found one yet.");
    tie_G_eccent_sliders_Chkbx.setFocusable(false);
    tie_G_eccent_sliders_Chkbx.setVerticalTextPosition(javax.swing.SwingConstants.BOTTOM);
    tie_G_eccent_sliders_Chkbx.addActionListener(new java.awt.event.ActionListener() {
      public void actionPerformed(java.awt.event.ActionEvent evt) {
        tie_G_eccent_sliders_ChkbxActionPerformed(evt);
      }
    });
    jToolBar1.add(tie_G_eccent_sliders_Chkbx);

    getContentPane().add(jToolBar1, java.awt.BorderLayout.PAGE_START);
    getContentPane().add(mainCSA_demoPanel1, java.awt.BorderLayout.CENTER);

    jMenuBar1.setMinimumSize(new java.awt.Dimension(63, 30));
    jMenuBar1.setPreferredSize(new java.awt.Dimension(74, 30));

    helpMenu.setText("About");
    helpMenu.setFont(new java.awt.Font("Segoe UI", 0, 14)); // NOI18N
    helpMenu.setMinimumSize(new java.awt.Dimension(63, 20));
    helpMenu.setPreferredSize(new java.awt.Dimension(63, 20));
    jMenuBar1.add(helpMenu);

    setJMenuBar(jMenuBar1);

    pack();
  }// </editor-fold>//GEN-END:initComponents

  private void instructionsBtnActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_instructionsBtnActionPerformed
    InstructionsFrame instructionsFrame = new InstructionsFrame();
    instructionsFrame.setSize(900, 800);
            
    instructionsFrame.setLocationRelativeTo(this);
    instructionsFrame.setLocation(400, 500);
    instructionsFrame.setVisible(true);
  }//GEN-LAST:event_instructionsBtnActionPerformed

  private void runExperimentBtnActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_runExperimentBtnActionPerformed
    RunExperimentDlg expDialog = new RunExperimentDlg(this, false);
    expDialog.setLocationRelativeTo(this);
    expDialog.setLocation(400, 500);
    expDialog.setVisible(true);
  }//GEN-LAST:event_runExperimentBtnActionPerformed

  private void showHoveringValsChkBxActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_showHoveringValsChkBxActionPerformed
    // TODO add your handling code here:
    ((MainCSA_demoPanel)mainCSA_demoPanel1).setShowHoveringVals(showHoveringValsChkBx.isSelected());
    ((MainCSA_demoPanel)mainCSA_demoPanel1).get_V_to_mu_Panel().repaint();
  }//GEN-LAST:event_showHoveringValsChkBxActionPerformed

  private void V_to_mu_fn_thickness_spinnerStateChanged(javax.swing.event.ChangeEvent evt) {//GEN-FIRST:event_V_to_mu_fn_thickness_spinnerStateChanged
    // TODO add your handling code here:
    ((MainCSA_demoPanel)mainCSA_demoPanel1).get_V_to_mu_Panel().setV_to_mu_fn_thickness((Integer)this.V_to_mu_fn_thickness_spinner.getValue());
    ((MainCSA_demoPanel)mainCSA_demoPanel1).get_V_to_mu_Panel().repaint();
  }//GEN-LAST:event_V_to_mu_fn_thickness_spinnerStateChanged

  private void jButton1ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton1ActionPerformed
    // TODO add your handling code here:
    Color newColor = JColorChooser.showDialog(null, "Choose a color", Color.black);
    ((MainCSA_demoPanel)mainCSA_demoPanel1).get_V_to_mu_Panel().V_to_mu_fn_color = newColor;
    // this.V_to_my_fn_thickness_spinner.setBackground(newColor);             // this doesn't seem to work. Rod 3-24-19
    ((MainCSA_demoPanel)mainCSA_demoPanel1).get_V_to_mu_Panel().repaint();
  }//GEN-LAST:event_jButton1ActionPerformed

  private void tie_G_eccent_sliders_ChkbxActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_tie_G_eccent_sliders_ChkbxActionPerformed
    // TODO add your handling code here:
    ((MainCSA_demoPanel)mainCSA_demoPanel1).setTie_G_V_ecc(tie_G_eccent_sliders_Chkbx.isSelected());
  }//GEN-LAST:event_tie_G_eccent_sliders_ChkbxActionPerformed

  /**
   * @param args the command line arguments
   */
  public static void main(String args[]) {
    /* Set the Nimbus look and feel */
    //<editor-fold defaultstate="collapsed" desc=" Look and feel setting code (optional) ">
    /* If Nimbus (introduced in Java SE 6) is not available, stay with the default look and feel.
         * For details see http://download.oracle.com/javase/tutorial/uiswing/lookandfeel/plaf.html 
     */
//    try {
//      for (javax.swing.UIManager.LookAndFeelInfo info : javax.swing.UIManager.getInstalledLookAndFeels()) {
//        if ("Nimbus".equals(info.getName())) {
//          javax.swing.UIManager.setLookAndFeel(info.getClassName());
//          break;
//        }
//      }
//    } catch (ClassNotFoundException ex) {
//      java.util.logging.Logger.getLogger(CSAdemo.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
//    } catch (InstantiationException ex) {
//      java.util.logging.Logger.getLogger(CSAdemo.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
//    } catch (IllegalAccessException ex) {
//      java.util.logging.Logger.getLogger(CSAdemo.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
//    } catch (javax.swing.UnsupportedLookAndFeelException ex) {
//      java.util.logging.Logger.getLogger(CSAdemo.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
//    }
    //</editor-fold>

    /* Create and display the form */
    java.awt.EventQueue.invokeLater(new Runnable() {
      public void run() {        
        new CSAdemo().setVisible(true);
      }
    });
  }
  
  public MainCSA_demoPanel getMain_CSA_panel()
  {
    return ((MainCSA_demoPanel)mainCSA_demoPanel1);
  }

  // Variables declaration - do not modify//GEN-BEGIN:variables
  private javax.swing.JSpinner V_to_mu_fn_thickness_spinner;
  private javax.swing.JMenu helpMenu;
  private javax.swing.JButton instructionsBtn;
  private javax.swing.JButton jButton1;
  private javax.swing.JLabel jLabel1;
  private javax.swing.JMenuBar jMenuBar1;
  private javax.swing.JToolBar.Separator jSeparator1;
  private javax.swing.JSeparator jSeparator2;
  private javax.swing.JToolBar.Separator jSeparator3;
  private javax.swing.JToolBar.Separator jSeparator4;
  private javax.swing.JToolBar.Separator jSeparator5;
  private javax.swing.JToolBar jToolBar1;
  private javax.swing.JPanel mainCSA_demoPanel1;
  private javax.swing.JButton runExperimentBtn;
  private javax.swing.JCheckBox showHoveringValsChkBx;
  private javax.swing.JCheckBox tie_G_eccent_sliders_Chkbx;
  // End of variables declaration//GEN-END:variables
}
