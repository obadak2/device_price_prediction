package org.example;

import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;

@Entity
public class Device {
    @Id
    @GeneratedValue(strategy=GenerationType.AUTO)
    private Integer id;

    private Integer battery_power;
    private Integer blue;
    private Double clock_speed;
    private Integer dual_sim;
    private Integer fc;
    private Integer four_g;
    private Integer int_memory;
    private Double m_dep;
    private Integer mobile_wt;
    private Integer n_cores;
    private Integer pc;
    private Integer px_height;
    private Integer px_width;
    private Integer ram;
    private Integer sc_h;
    private Integer sc_w;
    private Integer talk_time;
    private Integer three_g;
    private Integer touch_screen;
    private Integer wifi;


    public Integer getId() {
        return id;
    }

    public Integer getBattery_power() {
        return battery_power;
    }

    public Integer getBlue() {
        return blue;
    }

    public Double getClock_speed() {
        return clock_speed;
    }

    public Integer getDual_sim() {
        return dual_sim;
    }

    public Integer getFc() {
        return fc;
    }

    public Integer getFour_g() {
        return four_g;
    }

    public Integer getInt_memory() {
        return int_memory;
    }

    public Double getM_dep() {
        return m_dep;
    }

    public Integer getMobile_wt() {
        return mobile_wt;
    }

    public Integer getN_cores() {
        return n_cores;
    }

    public Integer getPc() {
        return pc;
    }

    public Integer getPx_height() {
        return px_height;
    }

    public Integer getPx_width() {
        return px_width;
    }

    public Integer getRam() {
        return ram;
    }

    public Integer getSc_h() {
        return sc_h;
    }

    public Integer getSc_w() {
        return sc_w;
    }

    public Integer getTalk_time() {
        return talk_time;
    }

    public Integer getThree_g() {
        return three_g;
    }

    public Integer getTouch_screen() {
        return touch_screen;
    }

    public Integer getWifi() {
        return wifi;
    }

    public Device() {
    }
}
