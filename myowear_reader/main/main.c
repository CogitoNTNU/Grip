#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_adc/adc_oneshot.h"
#include "esp_adc/adc_cali.h"
#include "esp_adc/adc_cali_scheme.h"

void app_main(void) {
    // 1) Init ADC unit
    adc_oneshot_unit_handle_t adc1;
    adc_oneshot_unit_init_cfg_t unit_cfg = { .unit_id = ADC_UNIT_1 };
    adc_oneshot_new_unit(&unit_cfg, &adc1);

    // 2) Konfigurer kanal: GPIO2 = ADC1_CHANNEL_1
    adc_oneshot_chan_cfg_t ch_cfg = {
        .bitwidth = ADC_BITWIDTH_DEFAULT,
        .atten    = ADC_ATTEN_DB_11    // ~0–3.3 V område
    };
    adc_oneshot_config_channel(adc1, ADC_CHANNEL_1, &ch_cfg);

    // 3) Kalibrering for mV
    adc_cali_handle_t cal;
    adc_cali_curve_fitting_config_t cal_cfg = {
        .unit_id = ADC_UNIT_1,
        .atten   = ADC_ATTEN_DB_11,
        .bitwidth= ADC_BITWIDTH_DEFAULT
    };
    bool calibrated = (adc_cali_create_scheme_curve_fitting(&cal_cfg, &cal) == ESP_OK);

    while (1) {
        int raw = 0, mv = 0;
        adc_oneshot_read(adc1, ADC_CHANNEL_1, &raw);
        if (calibrated) adc_cali_raw_to_voltage(cal, raw, &mv);
        printf("raw=%d, voltage=%d mV\n", raw, calibrated ? mv : -1);
        vTaskDelay(pdMS_TO_TICKS(20)); // ~50 Hz
    }
}
