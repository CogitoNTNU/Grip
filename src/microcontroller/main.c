/* 
 * File:   main.c
 * Author: afras
 *
 * Created on September 28, 2025, 12:13 AM
 */

#include <stdio.h>
#include <stdlib.h>
#include "sam.h"

#define MCLK_BASE       0x40000800
#define GCLK_BASE       0x40001C00

#define ADC0_BASE       0x43001C00  

#define PORT_BASE       0x41008000
#define PORTA_BASE      (PORT_BASE + 0)
#define PORTB_BASE      (PORT_BASE + 0x80)

#define SERCOM5_BASE    0x43000400

#define TC3_BASE        0x4101C000

#define MCLK_APBDMASK   *(volatile uint32_t*)(MCLK_BASE + 0x20) // ta baseadressen, legg til offset 0x20, behandle det som en peker til en 32-bit volatile int, og les/skriv verdien på den adressen?
#define MCLK_APBBMASK   *(volatile uint32_t*)(MCLK_BASE + 0x18)

#define GCLK_PCHCTRL40  (*(volatile uint32_t*)(GCLK_BASE + 0x80 + (40 * 4)))
#define GCLK_PCHCTRL35 (*(volatile uint32_t*)(GCLK_BASE + 0x80 + (35 * 4)))
#define GCLK_PCHCTRL26 (*(volatile uint32_t*)(GCLK_BASE + 0x80 + (26 * 4)))

#define ADC0_CTRLA      *(volatile uint16_t*)(ADC0_BASE + 0x00)
#define ADC0_INPUTCTRL  *(volatile uint16_t*)(ADC0_BASE + 0x04)
#define ADC0_INTFLAG    *(volatile uint8_t*)(ADC0_BASE + 0x2E)
#define ADC0_RESULT     *(volatile uint16_t*)(ADC0_BASE + 0x40)
#define ADC0_SWTRIG     *(volatile uint8_t*)(ADC0_BASE + 0x14)
#define ADC0_SYNCBUSY   *(volatile uint32_t*)(ADC0_BASE + 0x30)

#define PORTA_PINCFG3   *(volatile uint8_t*)(PORTA_BASE + 0x40 + 3)
#define PORTA_PMUX1     *(volatile uint8_t*)(PORTA_BASE + 0x30 + 1)

#define PORTA_PINCFG19  *(volatile uint8_t*)(PORTA_BASE + 0x40 + 19)
#define PORTA_PMUX9     *(volatile uint8_t*)(PORTA_BASE + 0x30 + 9)

#define PORTA_DIRSET   *(volatile uint32_t*)(PORTA_BASE + 0x08)
#define PORTA_OUTCLR   *(volatile uint32_t*)(PORTA_BASE + 0x14)
#define PORTA_OUTSET   *(volatile uint32_t*)(PORTA_BASE + 0x18)
#define PORTA_OUTTGL   *(volatile uint32_t*)(PORTA_BASE + 0x1C)
#define PORTA_PINCFG14 *(volatile uint8_t *)(PORTA_BASE + 0x40 + 14)

#define PORTB_PINCFG16  *(volatile uint8_t*)(PORTB_BASE + 0x40 + 16)
#define PORTB_PINCFG17  *(volatile uint8_t*)(PORTB_BASE + 0x40 + 17)
#define PORTB_PMUX8     *(volatile uint8_t*)(PORTB_BASE + 0x30 + 8)

#define SERCOM5_CTRLA       *(volatile uint32_t*)(SERCOM5_BASE + 0x00)
#define SERCOM5_CTRLB       *(volatile uint32_t*)(SERCOM5_BASE + 0x04)
#define SERCOM5_BAUD        *(volatile uint16_t*)(SERCOM5_BASE + 0x0C)
#define SERCOM5_DATA        *(volatile uint32_t*)(SERCOM5_BASE + 0x28)
#define SERCOM5_INTFLAG     *(volatile uint8_t*)(SERCOM5_BASE + 0x18)
#define SERCOM5_SYNCBUSY    *(volatile uint32_t*)(SERCOM5_BASE + 0x1C)

#define TC3_CTRLA       *(volatile uint32_t*)(TC3_BASE + 0x00)
#define TC3_CTRLBSET    *(volatile uint8_t*)(TC3_BASE + 0x05)
#define TC3_WAVE        *(volatile uint8_t*)(TC3_BASE + 0x0C)
#define TC3_CC0         *(volatile uint16_t*)(TC3_BASE + 0x1C)
#define TC3_CC1         *(volatile uint16_t*)(TC3_BASE + 0x1E)
#define TC3_SYNCBUSY    *(volatile uint32_t*)(TC3_BASE + 0x10)

void uart_send_char(char c) {
    while (!(SERCOM5_INTFLAG & (1 << 0)));            
    SERCOM5_DATA = c;
}

void uart_send_string(const char* str) {
    while (*str) {
        uart_send_char(*str++);
    }
}


void uart_send_adc(uint16_t value) {
    uart_send_char(0xAA);
    uart_send_char(value & 0xFF);
    uart_send_char((value >> 8) & 0xFF);
}

uint8_t uart_receive_char(void) {
    while (!(SERCOM5_INTFLAG & (1 << 2)));
    return (uint8_t)(SERCOM5_DATA & 0XFF);
}


uint8_t uart_data_available(void){
    return (SERCOM5_INTFLAG & (1 << 2));
}

void set_pwm_duty(uint16_t cc1)
{
    uint16_t top = TC3_CC0;
    if (cc1 > top) cc1 = top;
    TC3_CC1 = cc1;
    while (TC3_SYNCBUSY & (1 << 7));   // CC1 sync
}

static inline uint16_t adc_read_once(void) {
    ADC0_SWTRIG |= (1 << 1);
    while (!(ADC0_INTFLAG & (1 << 0)));
    uint16_t v = ADC0_RESULT;
    ADC0_INTFLAG = (1 << 0);
    return v;   
}

static inline void uart_send_packet(uint16_t val) {
    while (!(SERCOM5_INTFLAG & (1 << 0))); SERCOM5_DATA = 0XAA;
    while (!(SERCOM5_INTFLAG & (1 << 0))); SERCOM5_DATA = (uint8_t)(val & 0xFF);
    while (!(SERCOM5_INTFLAG & (1 << 0))); SERCOM5_DATA = (uint8_t)(val >> 8);   
}

static inline uint8_t uart_try_read(uint8_t* b) {
    if (SERCOM5_INTFLAG & (1 << 2)) {
        *b = (uint8_t)(SERCOM5_DATA & 0XFF);
        return 1;
    }
    return 0;
}

static inline void process_uart_rx_and_update_pwm(void) {
    static uint8_t st = 0;
    static uint16_t v = 0;
    uint8_t b;
    while (uart_try_read(&b)) {
        switch (st) {
            case 0: if (b == 0xAA) st = 1; break;
            case 1: v = b; st = 2; break;
            case 2: {
                v |= ((uint16_t)b) << 8;
                st = 0;
                uint16_t top = TC3_CC0;
                uint32_t tics = ((uint32_t)(top + 1) * v + 2047) / 4095;
                uint16_t cc1 = (tics > (top+1)) ? 0 : (uint16_t)(top - tics);
                TC3_CC1 = cc1;
                while (TC3_SYNCBUSY & (1 << 7));
                break;
            }
        }
    }
}

static inline void led_init_pa14(void) {
    PORTA_PINCFG14 &= ~(1<<0);        // PMUXEN=0 → ren GPIO
    PORTA_DIRSET    = (1<<14);        // PA14 som output
    PORTA_OUTSET    = (1<<14);        // LED av (aktiv-lav)
}

static inline void led_on(void)    { PORTA_OUTCLR = (1<<14); } // aktiv-lav
static inline void led_off(void)   { PORTA_OUTSET = (1<<14); }
static inline void led_toggle(void){ PORTA_OUTTGL = (1<<14); }


int main(int argc, char** argv) {
    led_init_pa14();
    led_on();   
    for (volatile int i=0;i<200000;i++);
    led_off();
    
    // Enable ADC0 og sercom5 klokke
    MCLK_APBDMASK |= (1 << 7) | (1 << 1);
    MCLK_APBBMASK |= (1 << 14);
    
    
    GCLK_PCHCTRL40 = (1 << 6) | (0 << 0);
    GCLK_PCHCTRL35 = (1 << 6) | (0 << 0); 
    GCLK_PCHCTRL26 = (1 << 6) | (0 << 0);

    
    // Config PA03 pin
    PORTA_PINCFG3 |= ((1 << 0) | (1 << 1));
    PORTA_PMUX1 = (PORTA_PMUX1 & 0x0F) | (0x1 << 4);
    
    // ADC setup
    ADC0_CTRLA |= (1 << 0);
    while (ADC0_SYNCBUSY & (1 << 0));
    
    ADC0_INPUTCTRL = (ADC0_INPUTCTRL & ~0x1F) | 0x01;
    while (ADC0_SYNCBUSY & (1 << 2));
    
    ADC0_CTRLA |= (1 << 1);
    while (ADC0_SYNCBUSY & (1 << 1));
    
    for(int i = 0; i < 100000; i++);

    // Configure PB16/PB17 for SERCOM   
    PORTB_PINCFG16  |= (1 << 0);
    PORTB_PINCFG17  |= (1 << 0);
    PORTB_PMUX8      = (0x2 << 4) | (0x2 << 0);
    
    
    
    // Uart
    SERCOM5_CTRLA = (1 << 0);
    while (SERCOM5_SYNCBUSY & (1 << 0));  
    
    SERCOM5_CTRLA    = (0x1 << 2) | (0 << 16) | (1 << 20) | (1 << 30);
    while (SERCOM5_SYNCBUSY & (1 << 1));
    
    SERCOM5_BAUD     = 0xF62B;
    
    SERCOM5_CTRLB = (1 << 16) | (1 << 17) | (0x0 << 0);
    while (SERCOM5_SYNCBUSY & (1 << 2));   
    
    SERCOM5_CTRLA   |= (1 << 1);    
    while (SERCOM5_SYNCBUSY & (1 << 1));
    
    
    PORTA_PINCFG19 |= (1 << 0);
    PORTA_PMUX9 = (PORTA_PMUX9 & 0X0F) | (0X4 << 4);
    
    TC3_CTRLA = (1 << 0);
    while (TC3_SYNCBUSY & (1 << 0));    
    TC3_CTRLA |= (0X00 << 2) | (0X0 << 4) | (0X0 << 8);    
    TC3_WAVE = 0X3;
    
    TC3_CC0 = 2399;  // Periode
    while (TC3_SYNCBUSY & (1 << 6));
    
    TC3_CC1 = 200;  // Start med 0% duty
    while (TC3_SYNCBUSY & (1 << 7));
    
    TC3_CTRLA |= (1 << 1);
    while (TC3_SYNCBUSY & (1 << 1));
    
    led_on();   
    for (volatile int i=0;i<20000000;i++);
    led_off();
    uart_send_string("boot\r\n");
    
    // read
    while (1) {
        led_on();   
        for (volatile int i=0;i<200000;i++);
        led_off();
        for (volatile int i=0;i<200000;i++);
//        uint16_t adc = adc_read_once();
//        uart_send_packet(adc);
//        process_uart_rx_and_update_pwm();
        uart_send_string("Kake \n");
    }
    

    return (0);
}

